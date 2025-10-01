import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Lasso

# importerer data
def import_data():
    model_ready_df = pd.read_csv("prosjekt/output/model_ready.csv")
    return model_ready_df

# deler data i mål- og prediktorvariabler og gjør om stations til å bruke one-hot encode
def split_data(model_ready_df):
    X = pd.get_dummies(model_ready_df.drop(columns=["free_bikes_next_hour", "timestamp"]),
                    columns=["station"], prefix="station")
    y = model_ready_df["free_bikes_next_hour"]

    # deler opp i trenings-, validerings- og testdata
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size=0.7, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size=0.5, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

# trener lasso-modell på ulike verdier av alpha
def train_lasso_model(X_train, X_val, X_test, y_train, y_val, y_test):
    lasso_models = {alpha: Lasso(alpha=alpha) for alpha in np.arange(0.01, 10, 0.1)}
    train_rmse_scores = {}
    val_rmse_scores = {}
    test_rmse_scores = {}

    for alpha, model in lasso_models.items():
        model.fit(X_train, y_train)
        train_rmse_scores[alpha] = root_mean_squared_error(y_train, model.predict(X_train))
        val_rmse_scores[alpha] = root_mean_squared_error(y_val, model.predict(X_val))
        test_rmse_scores[alpha] = root_mean_squared_error(y_test, model.predict(X_test))

    # finner hvilken alpha som gir laveste RMSE
    best_alpha = min(val_rmse_scores, key=val_rmse_scores.get)
    best_model = lasso_models[best_alpha]

    train_rmse = train_rmse_scores[best_alpha]
    val_rmse = val_rmse_scores[best_alpha]
    test_rmse = test_rmse_scores[best_alpha]

    return best_model, train_rmse, val_rmse, test_rmse