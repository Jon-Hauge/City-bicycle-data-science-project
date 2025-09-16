from data_ingest import load_data
from preprocessing import sort_timeframe, preprocess_data
from feature_engineering import merge_data
import os

# kjører pipelinen, som kombinerer alle andre filer og skriver til model_ready.csv
def run_pipeline(output_path="prosjekt/output/model_ready.csv"):
    print("Laster data...")
    stations_df, trips_df, weather_df = load_data()

    print("Preprosesserer data...")
    sort_timeframe(stations_df, trips_df, weather_df)
    target_stations_df, trips_hourly, trips_arrivals, trips_departures = preprocess_data(stations_df, trips_df)

    print("Lager features...")
    merged_df = merge_data(target_stations_df, trips_hourly, trips_arrivals, trips_departures, weather_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Ferdig! Lagret til {output_path}")


if __name__ == "__main__":
    run_pipeline()