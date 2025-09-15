import pandas as pd
import numpy as np

# Steg 1: Lesing av data

# leser inn data
stations_df = pd.read_csv("prosjekt/raw_data/stations.csv")
trips_df = pd.read_csv("prosjekt/raw_data/trips.csv")
weather_df = pd.read_csv("prosjekt/raw_data/weather.csv")

# konverterer til riktig tidssone - kan evt fjerne datetime
stations_df["timestamp"] = pd.to_datetime(stations_df["timestamp"]).dt.tz_convert("Europe/Oslo")
trips_df["started_at"] = pd.to_datetime(trips_df["started_at"], format="ISO8601").dt.tz_convert("Europe/Oslo")
trips_df["ended_at"] = pd.to_datetime(trips_df["ended_at"], format="ISO8601").dt.tz_convert("Europe/Oslo")
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"]).dt.tz_convert("Europe/Oslo")

# konverterer tidsstempel til hele timer og sorterer ihht LCOF
stations_df = stations_df.sort_values(["station", "timestamp"])
stations_df = stations_df.set_index("timestamp")
stations_df = stations_df.groupby("station")["free_bikes"].resample("h").ffill().reset_index()
