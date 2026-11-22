from datetime import datetime
import os
import sqlite3
from config import DB_NAME
from fetch_daily import run_fetch
from predict_today import run_prediction
from weather_stats import run_stats
from weather_train_model import run_training


def init_db() -> None:
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_weather (
                city TEXT NOT NULL,
                time TEXT NOT NULL,
                temperature_2m_max REAL,
                temperature_2m_min REAL,
                precipitation_sum REAL,
                wind_speed_10m_max REAL,
                surface_pressure_mean REAL,
                cloudcover_mean REAL,
                relative_humidity_2m_mean REAL,
                winddirection_10m_dominant REAL,
                PRIMARY KEY (city, time)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                forecast_date TEXT,
                city TEXT,
                predicted_temp REAL,
                actual_temp REAL,
                created_at TEXT,
                PRIMARY KEY (forecast_date, city)
            );
        """)
        conn.commit()


def main() -> None:
    init_db()

    print("Downloading yesterday's data")
    run_fetch()

    print("\nUpdating stats")
    run_stats()

    model_missing = not (
        os.path.exists("weather_lgbm_model.joblib")
        and os.path.exists("model_features.joblib")
    )
    if datetime.now().day in (1, 15) or model_missing:
        print("\nTraining model")
        run_training()

    print("\nToday's prediction")
    run_prediction()


if __name__ == "__main__":
    main()