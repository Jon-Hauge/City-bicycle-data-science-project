from data_utils import load_data, preprocess_data, merge_data
from train_model import split_data, train_lasso_pf_model
import pickle

def run_pipeline():
    # kombinerer alle filer til en ferdig prosessert model_ready_df
    print("----------------\nProsessering av data:\n----------------")

    print("Laster data...")
    stations_df, trips_df, weather_df = load_data()

    print("Preprosesserer data...")
    target_stations_df, trips_hourly, trips_arrivals, trips_departures = preprocess_data(stations_df, trips_df, weather_df)

    print("Lager features...")
    model_ready_df = merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df)

    print("\nFerdig!")

    # lagrer maskinlærings-modellen til egen fil og validerer
    print("\n---------------\nLagring av ML:\n---------------")
    
    print("Deler opp data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(model_ready_df)

    print("Finner beste modell og tilhørende RMSE...")
    best_model, train_rmse, val_rmse, test_rmse = train_lasso_pf_model(X_train, X_val, X_test, y_train, y_val, y_test)

    # lagrer beste modell og feature-kolonner separat
    output_path_ml = "prosjekt/models/best_model.pkl"
    pickle.dump(best_model, open("prosjekt/models/best_model.pkl", "wb"))
    feature_cols = list(X_train.columns)
    pickle.dump(feature_cols, open("prosjekt/models/feature_cols.pkl", "wb"))

    print(f"\nFerdig! Lagret til {output_path_ml}")

    finished_model = pickle.load(open("prosjekt/models/best_model.pkl", "rb"))
    print("\n-----------\nValidering:\n-----------")
    print("Modelltype:", type(finished_model))

    # Finner RMSE på data
    print("\nTreningsdata-RMSE-en er:", train_rmse,
        "\nValideringsdata-RMSE-en er:", val_rmse,
        "\nTestdata-RMSE-en er:", test_rmse)

if __name__ == "__main__":
    run_pipeline()