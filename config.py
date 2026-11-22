CITIES = {
    "Warszawa": {"latitude": 52.23, "longitude": 21.01},
    "Krakow": {"latitude": 50.06, "longitude": 19.94},
    "Lodz": {"latitude": 51.76, "longitude": 19.46},
    "Wroclaw": {"latitude": 51.11, "longitude": 17.03},
    "Poznan": {"latitude": 52.41, "longitude": 16.93},
    "Gdansk": {"latitude": 54.35, "longitude": 18.65},
    "Szczecin": {"latitude": 53.43, "longitude": 14.55},
    "Bydgoszcz": {"latitude": 53.12, "longitude": 18.01},
    "Torun": {"latitude": 53.01, "longitude": 18.60},
    "Lublin": {"latitude": 51.25, "longitude": 22.57},
    "Bialystok": {"latitude": 53.13, "longitude": 23.16},
    "Katowice": {"latitude": 50.26, "longitude": 19.02},
    "Kielce": {"latitude": 50.87, "longitude": 20.63},
    "Rzeszow": {"latitude": 50.04, "longitude": 22.00},
    "Olsztyn": {"latitude": 53.78, "longitude": 20.48},
    "Opole": {"latitude": 50.67, "longitude": 17.92},
    "Gorzow_Wielkopolski": {"latitude": 52.73, "longitude": 15.24},
    "Zielona_Gora": {"latitude": 51.94, "longitude": 15.51},
    "Siedlce": {"latitude": 52.17, "longitude": 22.28},
}

CITY_CATEGORIES = sorted(list(CITIES.keys()))

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

DB_NAME = "weather_forecast.db"

DAILIES = [ "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "surface_pressure_mean",
            "cloudcover_mean",
            "relative_humidity_2m_mean",
            "winddirection_10m_dominant" ]
