import numpy as np
import pandas as pd


# OpenAI's ChatGPT ble brukt til å formulere og/eller komme opp med ideer for deler av kode.
# Dette gjelder merge-funksjonene for å smelte sammen datasettene og interpolering
# og utfylling av NaN-rader til å inneholde 0 for analyse i del 2.
# All kode som er hjulpet av AI er forstått i seg selv og ihht til resten av programmet.
# OpenAI - https://chatgpt.com - henta 14.09.25 

def merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df):

    # slår sammen alt til en dataframe, merged_df, med alle stasjoner, sykkelbruk og vær
    merged_df = pd.merge_asof(target_stations_df.sort_values("timestamp"), trips_arrivals.sort_values("timestamp"),
                            by="station", left_on="timestamp", right_on="timestamp", direction="backward")
    merged_df = pd.merge_asof(merged_df.sort_values("timestamp"), trips_departures.sort_values("timestamp"),
                                        by="station", left_on="timestamp", right_on="timestamp", direction="backward")
    merged_df = merged_df.merge(weather_df, on="timestamp", how="left")
    merged_df = merged_df.merge(trips_hourly, on="timestamp", how="left")

    # legger til nye kolonner for blant annet prediksjonslabel og utleding av timestamp
    merged_df["free_bikes_next_hour"] = (merged_df.groupby("station")["free_bikes"].shift(-1))
    merged_df["hour"] = merged_df["timestamp"].dt.hour
    merged_df["day_of_week"] = merged_df["timestamp"].dt.day_of_week
    merged_df["month"] = merged_df["timestamp"].dt.month

    # dropper NaN-verdier og legger til 0 for å fylle rader i preparasjon for del 2
    merged_df = merged_df.dropna(subset=["free_bikes_next_hour"])
    merged_df["free_bikes"] = merged_df["free_bikes"].fillna(0).astype(int)
    merged_df["arrivals"] = merged_df["arrivals"].fillna(0).astype(int)
    merged_df["departures"] = merged_df["departures"].fillna(0).astype(int)
    merged_df["trips_count"] = merged_df["trips_count"].fillna(0).astype(int)
    merged_df["temperature"] = merged_df["temperature"].interpolate()
    merged_df["precipitation"] = merged_df["precipitation"].fillna(0).astype(int)
    merged_df["wind_speed"] = merged_df["wind_speed"].interpolate()

    # omsorterer kolonnene i merged_df
    cols = [
        "timestamp",
        "hour",
        "day_of_week",
        "month",
        "station",
        "free_bikes",
        "free_bikes_next_hour",
        "arrivals",
        "departures",
        "trips_count",
        "temperature",
        "precipitation",
        "wind_speed"
    ]
    merged_df = merged_df[cols]

    return merged_df