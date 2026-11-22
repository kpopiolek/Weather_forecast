import argparse
import re
import sqlite3
import time
import pandas as pd
import requests as r
from config import ARCHIVE_API_URL, CITIES, DB_NAME


def backfill_column(column_name: str, sql_type: str = "REAL") -> None:
    if not re.match(r"^[a-z0-9_]+$", column_name):
        raise ValueError(f"Incorrect column name: '{column_name}'")

    if sql_type not in {"REAL", "INTEGER", "TEXT"}:
        raise ValueError(f"Invalid SQL type: '{sql_type}'")

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        # 1. Sprawdzenie istniejących kolumn w tabeli
        cursor.execute("PRAGMA table_info(historical_weather);")
        existing_cols = [row[1] for row in cursor.fetchall()]

        if column_name not in existing_cols:
            cursor.execute(
                f"ALTER TABLE historical_weather ADD COLUMN {column_name} {sql_type};"
            )
            conn.commit()
            print(f"Added '{column_name}' to historical_weather_data.")
        else:
            print(f"Column '{column_name}' already exists – filling missing values.")

        # 2. Pobranie zakresu dat w bazie dla każdego miasta
        cursor.execute("""
            SELECT city, MIN(time), MAX(time)
            FROM historical_weather
            GROUP BY city
        """)
        city_ranges = cursor.fetchall()

        update_sql = f"""
            UPDATE historical_weather
            SET {column_name} = ?
            WHERE city = ? AND time = ?
        """

        for city, min_date, max_date in city_ranges:
            if city not in CITIES:
                continue

            print(f"Downloading '{column_name}' for: {city} ({min_date} -> {max_date})...")
            params = {
                "latitude": CITIES[city]["latitude"],
                "longitude": CITIES[city]["longitude"],
                "start_date": min_date,
                "end_date": max_date,
                "timezone": "Europe/Warsaw",
                "daily": [column_name],
            }

            try:
                res = r.get(ARCHIVE_API_URL, params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except (r.RequestException, ValueError) as e:
                print(f"[API ERROR] {city}: {e}")
                continue

            if "daily" not in data or column_name not in data["daily"]:
                print(f"[DATA ERROR] {city}: {data}")
                continue

            df_city = pd.DataFrame(data["daily"]).dropna(subset=[column_name])
            rows = [
                (row[column_name], city, row["time"])
                for _, row in df_city.iterrows()
            ]

            cursor.executemany(update_sql, rows)
            conn.commit()
            print(f"-> Updated {len(rows)} rows for {city}.")

            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Added new column form API to the database."
    )
    parser.add_argument(
        "--column",
        required=True,
        help="Full name of a column from OpenMeteo (eg. snowfall_sum)",
    )
    parser.add_argument(
        "--type",
        default="REAL",
        choices=["REAL", "INTEGER", "TEXT"],
        help="Column type SQLite (default: REAL)",
    )

    args = parser.parse_args()
    backfill_column(args.column, args.type)