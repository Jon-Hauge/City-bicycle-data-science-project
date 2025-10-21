import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

def split_data(model_ready_df):
    # deler data i mål- og prediktorvariabler og gjør om stations til å bruke one-hot encode
    X = pd.get_dummies(model_ready_df.drop(columns=["free_bikes_next_hour", "timestamp"]),
                    columns=["station"], prefix="station")
    y = model_ready_df["free_bikes_next_hour"]

    # deler opp i trenings-, validerings- og testdata uten tilfeldig stokking og med satt random_seed
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size=0.5, shuffle=False, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_lasso_pf_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # definerer polynom-transformerte data
    poly = PolynomialFeatures(degree=2)
    X_train_pf = poly.fit_transform(X_train)
    X_val_pf = poly.transform(X_val)
    X_test_pf = poly.transform(X_test)

    # lager og trener lasso-modeller på verdier av alpha mellom 0.01 og 10
    lasso_pf_models = {alpha: Lasso(alpha=alpha) for alpha in np.arange(0.01, 10, 0.1)}

    val_rmse_scores = {}

    for alpha, model in lasso_pf_models.items():
        model.fit(X_train_pf, y_train)
        val_rmse_scores[alpha] = root_mean_squared_error(y_val, model.predict(X_val_pf))

    # finner hvilken alpha som gir lavest RMSE på valideringsdata
    best_alpha = min(val_rmse_scores, key=val_rmse_scores.get)

    # finner modell for beste verdi av alpha
    best_model = lasso_pf_models[best_alpha]

    # finner RMSE på all data for beste verdi av alpha
    train_rmse = round(root_mean_squared_error(y_train, best_model.predict(X_train_pf)), 3)
    val_rmse = round(root_mean_squared_error(y_val, best_model.predict(X_val_pf)), 3)
    test_rmse = round(root_mean_squared_error(y_test, best_model.predict(X_test_pf)), 3)

    return best_model, train_rmse, val_rmse, test_rmse