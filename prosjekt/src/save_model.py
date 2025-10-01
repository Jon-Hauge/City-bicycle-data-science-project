import joblib
from train_model import import_data, split_data, train_lasso_model

# lagrer modellen til egen fil og validerer
def save_and_validate_model(output_path="prosjekt/models/best_model.pkl"):
    print("--------\nLagring:\n--------")

    print("Laster data...")
    model_ready_df = import_data()
    
    print("Deler opp data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(model_ready_df)

    print("Finner beste modell og tilhørende RMSE...")
    best_model, train_rmse, val_rmse, test_rmse = train_lasso_model(X_train, X_val, X_test, y_train, y_val, y_test)

    joblib.dump(best_model, "prosjekt/models/best_model.pkl")
    print(f"\nFerdig! Lagret til {output_path}")

    finished_model = joblib.load("prosjekt/models/best_model.pkl")
    print("\n-----------\nValidering:\n-----------")
    print("Modelltype:", type(finished_model))

    print("\nTreningsdata-RMSE-en er:", round(train_rmse, 3),
      "\nValideringsdata-RMSE-en er:", round(val_rmse, 3),
      "\nTestdata-RMSE-en er:", round(test_rmse, 3))
    
if __name__ == "__main__":
    save_and_validate_model()