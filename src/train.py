from data_utils import load_raw_data, preprocess_data, merge_data
from train_models import load_model_ready, split_data, train_models
import os
import pickle

def run_pipeline():
    # kombinerer alle filer til en ferdig prosessert model_ready_df
    print("----------------\nLagring av csv:\n----------------")

    print("Laster inn data...")
    stations_df, trips_df, weather_df = load_raw_data()

    print("Preprosesserer data...")
    target_stations_df, trips_hourly, trips_arrivals, trips_departures = preprocess_data(stations_df, trips_df, weather_df)

    print("Lager features...")
    model_ready_df = merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df)
    
    print("Lagrer csv-fil...")
    output_path_csv = "output/model_ready.csv"
    os.makedirs(os.path.dirname(output_path_csv), exist_ok=True)
    model_ready_df.to_csv(output_path_csv, index=False)

    print(f"\nFerdig! Lagret til {output_path_csv}")

    # lagrer maskinlærings-modellen til egen fil, validerer og finner RMSE på data
    print("\n---------------\nLagring av ML:\n---------------")

    print("Laster inn data...")
    model_ready_df = load_model_ready()
    
    print("Deler opp data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(model_ready_df)

    print("Finner beste modell og tilhørende RMSE...")
    best_model, train_rmse, val_rmse, test_rmse = train_models(X_train, X_val, X_test, y_train, y_val, y_test)

    output_path_ml = "models/best_model.pkl"
    output_path_fcl = "models/feature_cols.pkl"
    feature_cols = list(X_train.columns)

    pickle.dump(best_model, open(output_path_ml, "wb"))
    pickle.dump(feature_cols, open(output_path_fcl, "wb"))

    print(f"\nFerdig! Lagret modell til {output_path_ml}\n og features til {output_path_fcl}")

    finished_model = pickle.load(open("models/best_model.pkl", "rb"))
    print("\n-----------\nValidering:\n-----------")
    print("Modelltype:", type(finished_model))

    print("\nTreningsdata-RMSE-en er:", train_rmse,
        "\nValideringsdata-RMSE-en er:", val_rmse,
        "\nTestdata-RMSE-en er:", test_rmse)

if __name__ == "__main__":
    run_pipeline()