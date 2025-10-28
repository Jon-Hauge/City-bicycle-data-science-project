import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import PolynomialFeatures
import pickle

from data_utils import load_raw_data, preprocess_data, merge_data

def predict():
    # laster inn, sorterer, preprosesser og smelter sammen data
    stations_df, trips_df, weather_df = load_raw_data()
    target_stations_df, trips_hourly, trips_arrivals, trips_departures = preprocess_data(stations_df, trips_df, weather_df)
    model_ready_df = merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df)

    # finner siste timestamp i data, neste hele klokketime og predikert time.
    # Må bruke stations_df til siste timestamp siden merge_data runder opp.
    # Trekker da fra timer som fjernes fra model_ready_df i resampling
    last_timestamp = stations_df["timestamp"].max() #- timedelta(hours=2)
    next_hour = pd.to_datetime(model_ready_df["timestamp"].max()) + timedelta(hours=2)
    prediction_timestamp = next_hour + timedelta(hours=1)

    # laster inn ML-modellen og kolonnene med features, og definerer target_stations
    model = pickle.load(open("prosjekt/models/best_model.pkl", "rb"))
    feature_cols = pickle.load(open("prosjekt/models/feature_cols.pkl", "rb"))
    target_stations = ["Møllendalsplass", "Torgallmenningen", "Grieghallen",
                        "Høyteknologisenteret", "Studentboligene", "Akvariet",
                        "Damsgårdsveien 71", "Dreggsallmenningen Sør", "Florida Bybanestopp"]
    
    # legger sammen alle rader for prediksjon i model_ready_df, og kombinerer til en
    # dataframe med dummyvariabler for stasjonene
    rows = []
    for station in target_stations:
        last_row = model_ready_df[model_ready_df["station"]==station].iloc[-1:].drop(
            columns=["free_bikes_next_hour", "timestamp"]
        )
        last_row["station"] = station
        rows.append(last_row)

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df = pd.get_dummies(pred_df, columns=["station"], prefix="station")

    # fyller inn null i tomme kolonner og sørger for at prediksjons-dataframen
    # har riktige features
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0
    pred_df = pred_df[feature_cols]

    # polynomtransformerer datasettet med grad 2 og predikerer
    poly = PolynomialFeatures(degree=2)
    pred_df_pf = poly.fit_transform(pred_df)
    preds = model.predict(pred_df_pf)

    # samler resultatene i en array og printer dem
    results = []
    for i, station in enumerate(rows):
        station_name = station["station"].values[0]
        current_bikes = model_ready_df[model_ready_df["station"] == station_name]["free_bikes"].iloc[-1]
        results.append({
            "Stasjon": station_name,
            "Nåværende sykler": current_bikes,
            "Predikerte sykler": int(round(preds[i]))
        })

    results_df = pd.DataFrame(results)

    print(f"Siste tidsstempel i data: {last_timestamp}")
    print(f"Neste hele klokketime: {next_hour}")
    print(f"Predikerer for tidsstempel: {prediction_timestamp}\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    predict()