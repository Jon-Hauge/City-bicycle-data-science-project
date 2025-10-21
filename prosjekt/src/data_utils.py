import numpy as np
import pandas as pd

def load_data():

    # laster inn rådata
    stations_df = pd.read_csv("prosjekt/raw_data/stations.csv")
    trips_df = pd.read_csv("prosjekt/raw_data/trips.csv")
    weather_df = pd.read_csv("prosjekt/raw_data/weather.csv")

    return stations_df, trips_df, weather_df

# OpenAI's ChatGPT ble brukt til å formulere og/eller komme opp med ideer for deler av kode.
# Dette gjelder target_stations_df, der det blir brukt resampling ihht til LOCF-prinsippet
# og opprettelsen av de tre datasettene fra trips.
# All kode som er hjulpet av AI er forstått i seg selv og ihht til resten av programmet.
# OpenAI - https://chatgpt.com - henta 14.09.25

def preprocess_data(stations_df, trips_df, weather_df):

    # konverterer timestamp på all data til riktig format og tidssone
    stations_df["timestamp"] = pd.to_datetime(stations_df["timestamp"], utc=True).dt.tz_convert("Europe/Oslo")
    trips_df["started_at"] = pd.to_datetime(trips_df["started_at"], utc=True, format="ISO8601").dt.tz_convert("Europe/Oslo")
    trips_df["ended_at"] = pd.to_datetime(trips_df["ended_at"], utc=True,  format="ISO8601").dt.tz_convert("Europe/Oslo")
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True).dt.tz_convert("Europe/Oslo")

    # Oppretter en ny dataframe med kun timestamp, target_stations og
    # ledige sykler utifra stations_df. Sorterer også timer ihht LOCF
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

    # preprosesserer trips ved å lage dataframes for totalt antall turer
    # og ankomst og avreise fra stasjon rundet til neste hele time
    trips_hourly = trips_df.groupby(trips_df["started_at"].dt.floor("h")).size().reset_index(name="trips_count")
    trips_hourly = trips_hourly.rename(columns={"started_at": "timestamp"})

    trips_arrivals = trips_df.groupby([trips_df["ended_at"].dt.floor("h"), "end_station_name"]).size().reset_index(name="arrivals")
    trips_arrivals = trips_arrivals.rename(columns={"end_station_name":"station", "ended_at":"timestamp"})

    trips_departures = trips_df.groupby([trips_df["started_at"].dt.floor("h"), "start_station_name"]).size().reset_index(name="departures")
    trips_departures = trips_departures.rename(columns={"start_station_name":"station", "started_at":"timestamp"})

    return target_stations_df, trips_hourly, trips_arrivals, trips_departures



# OpenAI's ChatGPT ble brukt til å formulere og/eller komme opp med ideer for deler av kode.
# Dette gjelder merge-funksjonene for å smelte sammen datasettene og interpolering
# og utfylling av NaN-rader til å inneholde 0 for analyse og modelltrening
# All kode som er hjulpet av AI er forstått i seg selv og ihht til resten av programmet.
# OpenAI - https://chatgpt.com - henta 14.09.25 

def merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df):

    # slår sammen alt til en felles dataframe, merged_df, med alle stasjoner, sykkelbruk og vær.
    # Sorteres på timestamp fra target_stations_df, så får kun målinger fra relevant tidsperiode
    merged_df = pd.merge_asof(target_stations_df.sort_values("timestamp"), trips_arrivals.sort_values("timestamp"),
                            by="station", left_on="timestamp", right_on="timestamp", direction="backward")
    merged_df = pd.merge_asof(merged_df.sort_values("timestamp"), trips_departures.sort_values("timestamp"),
                                        by="station", left_on="timestamp", right_on="timestamp", direction="backward")
    merged_df = merged_df.merge(weather_df, on="timestamp", how="left")
    merged_df = merged_df.merge(trips_hourly, on="timestamp", how="left")

    # legger til nye kolonner for prediksjonslabel og utleding av timestamp til modelltrening
    merged_df["free_bikes_next_hour"] = (merged_df.groupby("station")["free_bikes"].shift(-1))
    merged_df["hour"] = merged_df["timestamp"].dt.hour
    merged_df["day_of_week"] = merged_df["timestamp"].dt.day_of_week
    merged_df["month"] = merged_df["timestamp"].dt.month

    # interpolerer temperatur og vindfart og fyller andre NaN-verdier med 0
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

    # dropper målinger fra 15.04.24 - 15.08.24 grunnet statiske målinger i perioden
    mask = ~((merged_df["timestamp"] >= "2024-04-15") & (merged_df["timestamp"] <= "2024-08-15"))
    merged_df = merged_df[mask]


    return merged_df