import pandas as pd

# leser inn data
def load_data():
    stations_df = pd.read_csv("prosjekt/raw_data/stations.csv")
    trips_df = pd.read_csv("prosjekt/raw_data/trips.csv")
    weather_df = pd.read_csv("prosjekt/raw_data/weather.csv")

    return stations_df, trips_df, weather_df