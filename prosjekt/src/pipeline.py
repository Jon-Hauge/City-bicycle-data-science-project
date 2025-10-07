from data_ingest import load_data
from preprocessing import sort_timeframe, preprocess_data
from feature_engineering import merge_data
from train_model import import_model, split_data, train_lasso_model
import pickle
import os

def run_pipeline():
    # kombinerer alle filer og skriver til model_ready.csv
    print("----------------\nLagring av csv:\n----------------")

    print("Laster data...")
    stations_df, trips_df, weather_df = load_data()

    print("Preprosesserer data...")
    sort_timeframe(stations_df, trips_df, weather_df)
    target_stations_df, trips_hourly, trips_arrivals, trips_departures = preprocess_data(stations_df, trips_df)

    print("Lager features...")
    merged_df = merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df)

    output_path_csv = "prosjekt/output/model_ready.csv"
    os.makedirs(os.path.dirname(output_path_csv), exist_ok=True)
    merged_df.to_csv(output_path_csv, index=False)
    print(f"\nFerdig! Lagret til {output_path_csv}")

    
    # lagrer maskinlærings-modellen til egen fil og validerer
    print("\n---------------\nLagring av ML:\n---------------")

    print("Laster data...")
    model_ready_df = import_model()
    
    print("Deler opp data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(model_ready_df)

    print("Finner beste modell og tilhørende RMSE...")
    best_model, train_rmse, val_rmse, test_rmse = train_lasso_model(X_train, X_val, X_test, y_train, y_val, y_test)

    output_path_ml = "prosjekt/models/best_model.pkl"
    pickle.dump(best_model, open("prosjekt/models/best_model.pkl", "wb"))
    print(f"\nFerdig! Lagret til {output_path_ml}")

    finished_model = pickle.load(open("prosjekt/models/best_model.pkl", "rb"))
    print("\n-----------\nValidering:\n-----------")
    print("Modelltype:", type(finished_model))

    print("\nTreningsdata-RMSE-en er:", round(train_rmse, 3),
      "\nValideringsdata-RMSE-en er:", round(val_rmse, 3),
      "\nTestdata-RMSE-en er:", round(test_rmse, 3))

if __name__ == "__main__":
    run_pipeline()