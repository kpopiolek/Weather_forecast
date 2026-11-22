from datetime import datetime, timedelta
import sqlite3
import time
import pandas as pd
import requests as r
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from config import ARCHIVE_API_URL, CITIES, DAILIES, DB_NAME

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"


def _save_daily_df(
    conn: sqlite3.Connection, df_city: pd.DataFrame, city: str, max_date: str
) -> int:
    missing = set(DAILIES) - set(df_city.columns)
    if missing:
        print(f"[MISSING COLUMNS] {city}: {sorted(missing)}")
        return 0

    df_city = df_city[df_city["time"] <= max_date].copy()
    df_city["city"] = city
    df_city = df_city.dropna(subset=DAILIES)

    if df_city.empty:
        return 0

    cols = ["city", "time"] + DAILIES
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)

    insert_sql = f"""
        INSERT OR IGNORE INTO historical_weather ({col_names})
        VALUES ({placeholders})
    """
    rows = list(df_city[cols].itertuples(index=False, name=None))
    cursor = conn.cursor()
    cursor.executemany(insert_sql, rows)
    conn.commit()
    return len(rows)


def run_fetch() -> None:
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    archive_cutoff = (today - timedelta(days=5)).strftime("%Y-%m-%d")

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT city, MAX(time) as last_date 
            FROM historical_weather 
            GROUP BY city
        """)
        db_dates = dict(cursor.fetchall())

        to_fetch = {}
        for city in CITIES.keys():
            last_date = db_dates.get(city)
            if last_date is None:
                to_fetch[city] = "2009-01-01"
            elif last_date < yesterday:
                next_day = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                to_fetch[city] = next_day.strftime("%Y-%m-%d")

        if not to_fetch:
            print("All cities are up to date.")
            return

        print("To fetch:", to_fetch)

        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = r.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        for city, start_date in to_fetch.items():
            saved_total = 0

            if start_date <= archive_cutoff:
                archive_params = {
                    "latitude": CITIES[city]["latitude"],
                    "longitude": CITIES[city]["longitude"],
                    "start_date": start_date,
                    "end_date": archive_cutoff,
                    "timezone": "Europe/Warsaw",
                    "daily": DAILIES,
                }
                try:
                    res = session.get(ARCHIVE_API_URL, params=archive_params, timeout=30)
                    res.raise_for_status()
                    data = res.json()
                    if "daily" not in data:
                        print(f"[ARCHIVE DATA ERROR] {city}: {data}")
                        continue
                    df_arch = pd.DataFrame(data["daily"])
                    saved_total += _save_daily_df(conn, df_arch, city, archive_cutoff)

                    if (
                        datetime.strptime(archive_cutoff, "%Y-%m-%d")
                        - datetime.strptime(start_date, "%Y-%m-%d")
                    ).days > 365:
                        time.sleep(31)
                except (r.RequestException, ValueError) as e:
                    print(f"[ARCHIVE API ERROR] {city}: {e} - skipping recent fetch to preserve history.")
                    continue

            forecast_dailies = [
                "cloud_cover_mean" if col == "cloudcover_mean" else col
                for col in DAILIES
            ]
            forecast_params = {
                "latitude": CITIES[city]["latitude"],
                "longitude": CITIES[city]["longitude"],
                "past_days": 7,
                "forecast_days": 0,
                "timezone": "Europe/Warsaw",
                "daily": forecast_dailies,
            }
            try:
                res = session.get(FORECAST_API_URL, params=forecast_params, timeout=15)
                res.raise_for_status()
                data = res.json()
                if "daily" in data:
                    df_recent = pd.DataFrame(data["daily"]).rename(
                        columns={"cloud_cover_mean": "cloudcover_mean"}
                    )
                    saved_total += _save_daily_df(conn, df_recent, city, yesterday)
                else:
                    print(f"[FORECAST DATA ERROR] {city}: {data}")
            except (r.RequestException, ValueError) as e:
                print(f"[FORECAST API ERROR] {city}: {e}")

            print(f"Updated {city}: saved {saved_total} rows (up to {yesterday})")

    print("Finished fetching daily weather.")


if __name__ == "__main__":
    run_fetch()