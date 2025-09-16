import numpy as np
import pandas as pd

def sort_timeframe(stations_df, trips_df, weather_df):
    # konverterer timestamp på all data til riktig format og tidssone
    # Filtrerer også data i trips og weather til å kun omhandle perioden stations måler
    stations_df["timestamp"] = pd.to_datetime(stations_df["timestamp"], utc=True).dt.tz_convert("Europe/Oslo")
    trips_df["started_at"] = pd.to_datetime(trips_df["started_at"], utc=True, format="ISO8601").dt.tz_convert("Europe/Oslo")
    trips_df["ended_at"] = pd.to_datetime(trips_df["ended_at"], utc=True,  format="ISO8601").dt.tz_convert("Europe/Oslo")
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True).dt.tz_convert("Europe/Oslo")

    start_time = stations_df["timestamp"].min()
    end_time = stations_df["timestamp"].max()

    trips_df = trips_df[(trips_df["started_at"] >= start_time) & (trips_df["started_at"] <= end_time)]
    trips_df = trips_df[(trips_df["ended_at"] >= start_time) & (trips_df["ended_at"] <= end_time)]
    weather_df = weather_df[(weather_df["timestamp"] >= start_time) & (weather_df["timestamp"] <= end_time)]


# OpenAI's ChatGPT ble brukt til å formulere og/eller komme opp med ideer for deler av kode.
# Dette gjelder target_stations_df, der det blir brukt resampling ihht til LOCF-prinsippet
# og opprettelsen av de tre datasettene fra trips.
# All kode som er hjulpet av AI er forstått i seg selv og ihht til resten av programmet.
# OpenAI - https://chatgpt.com - henta 14.09.25

def preprocess_data(stations_df, trips_df):
    # preprosesserer stations ved å opprette ny dataframe med kun timestamp, target_stations
    # og ledige sykler. Sorterer i tillegg timer ihht LOCF
    target_stations = np.array(["Møllendalsplass", "Torgallmenningen", "Grieghallen",
                                "Høyteknologisenteret", "Studentboligene", "Akvariet",
                                "Damsgårdsveien 71", "Dreggsallmenningen Sør", "Florida Bybanestopp"])

    filtered_stations = stations_df[stations_df["station"].isin(target_stations)]
    filtered_stations = filtered_stations.set_index("timestamp").sort_index()
    target_stations_df = filtered_stations.groupby("station")["free_bikes"].apply(lambda s: s.resample("1h").ffill()
                                            ).reset_index().pivot(index="timestamp", columns="station",
                                            values="free_bikes").sort_index()
    target_stations_df.columns.name = None
    target_stations_df = target_stations_df.reset_index()

    target_stations_df = target_stations_df.melt(id_vars=["timestamp"], var_name="station", value_name="free_bikes")

    # preprosesserer trips ved å lage dataframes for totalt
    # antall turer og ankomst og avreise fra stasjon
    trips_hourly = trips_df.groupby(trips_df["started_at"].dt.floor("h")).size().reset_index(name="trips_count")
    trips_hourly = trips_hourly.rename(columns={"started_at": "timestamp"})

    trips_arrivals = trips_df.groupby([trips_df["ended_at"].dt.floor("h"), "end_station_name"]).size().reset_index(name="arrivals") #(["end_station_name", "ended_at"]).size().reset_index(name="arrivals")
    trips_arrivals = trips_arrivals.rename(columns={"end_station_name":"station", "ended_at":"timestamp"})

    trips_departures = trips_df.groupby([trips_df["started_at"].dt.floor("h"), "start_station_name"]).size().reset_index(name="departures")#(["start_station_name", "started_at"]).size().reset_index(name="departures")
    trips_departures = trips_departures.rename(columns={"start_station_name":"station", "started_at":"timestamp"})

    return target_stations_df, trips_hourly, trips_arrivals, trips_departures