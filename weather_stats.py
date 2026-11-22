import os
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DB_NAME


def run_stats() -> None:
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        update_query = """
            UPDATE predictions
            SET actual_temp = (
                SELECT h.temperature_2m_max
                FROM historical_weather h
                WHERE h.city = predictions.city 
                  AND h.time = predictions.forecast_date
            )
            WHERE actual_temp IS NULL 
              AND EXISTS (
                SELECT 1 FROM historical_weather h
                WHERE h.city = predictions.city 
                  AND h.time = predictions.forecast_date
            );
        """
        cursor.execute(update_query)
        updated_count = cursor.rowcount
        conn.commit()
        print(f"Synchronized temperatures: {updated_count}")

        stats_query = """
            SELECT 
                COUNT(*) AS evaluated_days,
                ROUND(AVG(ABS(actual_temp - predicted_temp)), 2) AS mae,
                AVG((actual_temp - predicted_temp) * (actual_temp - predicted_temp)) AS mse,
                ROUND(100.0 * SUM(CASE WHEN ABS(actual_temp - predicted_temp) <= 1.0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS hit_rate_1deg_pct,
                ROUND(100.0 * SUM(CASE WHEN ABS(actual_temp - predicted_temp) <= 2.0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS hit_rate_2deg_pct
            FROM predictions
            WHERE actual_temp IS NOT NULL;
        """
        df_global = pd.read_sql(stats_query, conn)

        if df_global["evaluated_days"].iloc[0] == 0:
            print("\nUnsynchronized predictions - skipping stats and plot.")
            return

        df_global["rmse"] = np.round(np.sqrt(df_global["mse"]), 2)
        df_global = df_global.drop(columns=["mse"])

        city_stats_query = """
            SELECT 
                city,
                COUNT(*) AS samples,
                ROUND(AVG(ABS(actual_temp - predicted_temp)), 2) AS mae,
                ROUND(100.0 * SUM(CASE WHEN ABS(actual_temp - predicted_temp) <= 1.0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS hit_rate_1deg_pct
            FROM predictions
            WHERE actual_temp IS NOT NULL
            GROUP BY city
            ORDER BY mae ASC;
        """
        df_by_city = pd.read_sql(city_stats_query, conn)

        print("\nGLOBAL STATS")
        print(df_global.to_string(index=False))

        print("\nSTATS PER CITY")
        print(df_by_city.to_string(index=False))

        plot_query = """
            SELECT city, predicted_temp, actual_temp, forecast_date
            FROM predictions
            WHERE actual_temp IS NOT NULL
              AND forecast_date = (SELECT MAX(forecast_date) FROM predictions WHERE actual_temp IS NOT NULL)
            ORDER BY actual_temp DESC;
        """
        df_plot = pd.read_sql(plot_query, conn)

    if not df_plot.empty:
        date_str = df_plot["forecast_date"].iloc[0]

        plt.figure(figsize=(14, 6))
        x = np.arange(len(df_plot))
        width = 0.35

        plt.bar(x - width / 2, df_plot["actual_temp"], width, label="Real (°C)", color="#2b5c8f")
        plt.bar(x + width / 2, df_plot["predicted_temp"], width, label="Prediction (°C)", color="#e26d5c")

        plt.xlabel("City")
        plt.ylabel("Max temperature (°C)")
        plt.title(f"Prediction and real value comparison ({date_str})")
        plt.xticks(x, df_plot["city"], rotation=45, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"comparison_{date_str}.png")

        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nSaved comparison plot to: {plot_path}")


if __name__ == "__main__":
    run_stats()