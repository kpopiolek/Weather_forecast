import datetime as dt
import sqlite3
import joblib
import numpy as np
import pandas as pd
from config import CITIES, DAILIES, DB_NAME


def run_prediction() -> None:
    model = joblib.load("weather_lgbm_model.joblib")
    features = joblib.load("model_features.joblib")

    today = dt.datetime.now()
    yesterday_str = (today - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    seven_days_ago_str = (today - dt.timedelta(days=7)).strftime("%Y-%m-%d")

    forecast_date = today.strftime("%Y-%m-%d")
    created_at = today.strftime("%Y-%m-%d %H:%M:%S")

    days = today.timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * days / 365.25)
    cos_day = np.cos(2 * np.pi * days / 365.25)

    input_rows = []

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                forecast_date TEXT,
                city TEXT,
                predicted_temp REAL,
                actual_temp REAL,
                created_at TEXT,
                PRIMARY KEY (forecast_date, city)
            )
        """)

        query = """
            SELECT * FROM historical_weather 
            WHERE time BETWEEN ? AND ? 
            ORDER BY city, time
        """
        all_weather = pd.read_sql(
            query, conn, params=(seven_days_ago_str, yesterday_str)
        )

        for city in CITIES.keys():
            weather = (
                all_weather[all_weather["city"] == city]
                .sort_values("time")
                .reset_index(drop=True)
            )

            if (
                len(weather) != 7
                or weather.iloc[-1]["time"] != yesterday_str
                or weather.iloc[0]["time"] != seven_days_ago_str
                or weather[DAILIES].isna().any().any()
            ):
                latest_date = weather.iloc[-1]["time"] if len(weather) > 0 else "brak"
                print(
                    f"[SKIPPED] {city}: incomplete data "
                    f"(found {len(weather)}/7 days, newest: {latest_date})."
                )
                continue

            yesterday_row = weather.iloc[-1]
            two_days_ago_row = weather.iloc[-2]

            temp_rolling_mean = weather["temperature_2m_max"].mean()
            temp_max_diff = (
                yesterday_row["temperature_2m_max"]
                - two_days_ago_row["temperature_2m_max"]
            )
            temp_amplitude_lag1 = (
                yesterday_row["temperature_2m_max"]
                - yesterday_row["temperature_2m_min"]
            )

            wind_rad = np.deg2rad(yesterday_row["winddirection_10m_dominant"])
            wind_u_lag1 = -yesterday_row["wind_speed_10m_max"] * np.sin(wind_rad)
            wind_v_lag1 = -yesterday_row["wind_speed_10m_max"] * np.cos(wind_rad)

            input_dict = {
                "city": city,
                "temp_max_rolling_mean_7": temp_rolling_mean,
                "temperature_2m_max_lag1": yesterday_row["temperature_2m_max"],
                "temperature_2m_min_lag1": yesterday_row["temperature_2m_min"],
                "precipitation_sum_lag1": yesterday_row["precipitation_sum"],
                "wind_speed_10m_max_lag1": yesterday_row["wind_speed_10m_max"],
                "surface_pressure_mean_lag1": yesterday_row["surface_pressure_mean"],
                "cloudcover_mean_lag1": yesterday_row["cloudcover_mean"],
                "relative_humidity_2m_mean_lag1": (
                    yesterday_row["relative_humidity_2m_mean"]
                ),
                "winddirection_10m_dominant_lag1": (
                    yesterday_row["winddirection_10m_dominant"]
                ),
                "temperature_2m_max_lag2": two_days_ago_row["temperature_2m_max"],
                "temperature_2m_min_lag2": two_days_ago_row["temperature_2m_min"],
                "temp_max_diff": temp_max_diff,
                "sin_day": sin_day,
                "cos_day": cos_day,
                "wind_v_lag1": wind_v_lag1,
                "wind_u_lag1": wind_u_lag1,
                "temp_amplitude_lag1": temp_amplitude_lag1,
            }
            input_rows.append(input_dict)

        if not input_rows:
            print("\nUncompete data for every city.")
            return

        X_today = pd.DataFrame(input_rows)[features]
        X_today["city"] = pd.Categorical(
            X_today["city"], categories=sorted(CITIES.keys())
        )

        preds = model.predict(X_today)

        for city, pred in zip(X_today["city"], preds):
            pred_val = round(float(pred), 2)
            cursor.execute(
                """
                    INSERT OR REPLACE INTO predictions (forecast_date, city, predicted_temp, actual_temp, created_at)
                    VALUES (?, ?, ?, (SELECT actual_temp FROM predictions WHERE forecast_date = ? AND city = ?), ?)
                """,
                (forecast_date, city, pred_val, forecast_date, city, created_at),
            )
            print(f"{city:20}: {pred_val:5.2f}°C")

        conn.commit()
        print(f"\nSaved predictions for date: {forecast_date}")


if __name__ == "__main__":
    run_prediction()