import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR


def load_model_ready():

    # laster inn ferdig preprosessert og sammensmeltet rådata
    model_ready_df = pd.read_csv("prosjekt/output/model_ready.csv")
    
    return model_ready_df

def split_data(model_ready_df):

    # deler data i mål- og prediktorvariabler og gjør om stations til å bruke one-hot encode
    X = pd.get_dummies(model_ready_df.drop(columns=["free_bikes_next_hour", "timestamp"]),
                    columns=["station"], prefix="station")
    y = model_ready_df["free_bikes_next_hour"]

    # deler opp i trenings-, validerings- og testdata uten tilfeldig stokking og med satt random_seed
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size=0.7, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size=0.5, shuffle=False, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, X_val, X_test, y_train, y_val, y_test):
    
    # definerer polynom-transformerte data for lasso_pf
    poly = PolynomialFeatures(degree=2)
    X_train_pf = poly.fit_transform(X_train)
    X_val_pf = poly.transform(X_val)
    X_test_pf = poly.transform(X_test)

    # lager og trener lasso-modeller på verdier av alpha mellom 0.01 og 10 med steg 0.1
    lasso_models = {alpha: Lasso(alpha=alpha) for alpha in np.arange(0.01, 10, 0.1)}
    lasso_models_pf = {alpha: Lasso(alpha=alpha) for alpha in np.arange(0.01, 10, 0.1)}

    # definerer dictionaries for å holde på validerings-RMSE for lasso-modellene
    val_rmse_scores_lasso = {}
    val_rmse_scores_lasso_pf = {}

    # trener lasso-modellene på ulike verdier av alpha og lagrer validerings-RMSE
    for alpha, model in lasso_models.items():
        model.fit(X_train, y_train)
        val_rmse_scores_lasso[alpha] = root_mean_squared_error(y_val, model.predict(X_val))
    
    for alpha, model in lasso_models_pf.items():
        model.fit(X_train_pf, y_train)
        val_rmse_scores_lasso_pf[alpha] = root_mean_squared_error(y_val, model.predict(X_val_pf))

    # finner hvilken alpha som gir lavest RMSE på valideringsdata
    best_alpha = min(val_rmse_scores_lasso, key=val_rmse_scores_lasso.get)
    best_alpha_pf = min(val_rmse_scores_lasso_pf, key=val_rmse_scores_lasso_pf.get)

    # finner modell for beste verdi av alpha
    best_model_lasso = lasso_models[best_alpha]
    best_model_lasso_pf = lasso_models_pf[best_alpha_pf]

    # legger til beste lasso-modeller i en dictionary og sammenligner med baseline og SVR
    models = {"Lasso": best_model_lasso,
          "Poly + Lasso": best_model_lasso_pf,
          "Support Vector": make_pipeline(StandardScaler(), SVR(kernel="linear")),
          "Baseline": DummyRegressor(strategy="mean")}
    
    # Trener kun SVR og baseline siden lasso-modellene allerede er trent
    models["Support Vector"].fit(X_train, y_train)
    models["Baseline"].fit(X_train, y_train)

    # sjekker RMSE på valideringsdata for alle modellene og velger ut beste modell
    curr_val_rmse = 0
    best_val_rmse = 10
    best_model_name = ""
    best_model = ""

    for name, model in models.items():
        if name == "Poly + Lasso":
            curr_val_rmse = root_mean_squared_error(y_val, model.predict(X_val_pf))
        else:
            curr_val_rmse = root_mean_squared_error(y_val, model.predict(X_val))

        if (curr_val_rmse < best_val_rmse):
            best_val_rmse = curr_val_rmse
            best_model_name = name
            best_model = model

    # finner RMSE på all data for beste modell og returnerer
    if best_model_name == "Poly + Lasso":
        train_rmse = round(root_mean_squared_error(y_train, best_model.predict(X_train_pf)), 3)
        val_rmse = round(root_mean_squared_error(y_val, best_model.predict(X_val_pf)), 3)
        test_rmse = round(root_mean_squared_error(y_test, best_model.predict(X_test_pf)), 3)
    else:
        train_rmse = round(root_mean_squared_error(y_train, best_model.predict(X_train)), 3)
        val_rmse = round(root_mean_squared_error(y_val, best_model.predict(X_val)), 3)
        test_rmse = round(root_mean_squared_error(y_test, best_model.predict(X_test)), 3)

    return best_model, train_rmse, val_rmse, test_rmse
