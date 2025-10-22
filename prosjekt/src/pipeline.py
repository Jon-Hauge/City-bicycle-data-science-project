from data_utils import load_raw_data, preprocess_data, merge_data
from train_model import split_data, train_lasso_pf_model
import os
import pickle

def run_pipeline():
    print("----------------\nLagring av csv:\n----------------")

    # laster inn, preprosesserer og smelter sammen rådata
    stations_df, trips_df, weather_df = load_raw_data()
    target_stations_df, trips_hourly, trips_arrivals, trips_departures = preprocess_data(stations_df, trips_df, weather_df)
    model_ready_df = merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df)

    # lagrer til egen csv-fil, model_ready.csv, via os
    output_path_csv = "prosjekt/output/model_ready.csv"
    os.makedirs(os.path.dirname(output_path_csv), exist_ok=True)
    model_ready_df.to_csv(output_path_csv, index=False)

    print(f"\nFerdig! Lagret til {output_path_csv}")

    print("\n---------------\nLagring av ML:\n---------------")

    # splitter opp i trenings-, validerings- og testdata og finner beste modell
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(model_ready_df)
    best_model,_,_,_ = train_lasso_pf_model(X_train, X_val, X_test, y_train, y_val, y_test)

    # lagrer til egen pkl-fil, best_model.pkl, via pickle
    output_path_ml = "prosjekt/models/best_model.pkl"
    pickle.dump(best_model, open(output_path_ml , "wb"))
    
    # lagrer også feature kolonner fra treningsdata til feature_cols.pkl
    # for å kunne hente valgte stasjoner i predikering
    output_path_fc = "prosjekt/models/feature_cols.pkl"
    feature_cols = list(X_train.columns)
    pickle.dump(feature_cols, open(output_path_fc, "wb"))

    print(f"\nFerdig! Lagret modell til {output_path_ml} \nog features til {output_path_fc}\n")

if __name__ == "__main__":
    run_pipeline()