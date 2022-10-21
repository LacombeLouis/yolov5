import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import QuantileRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings("ignore")

quantile_estimator_params = {
        "GradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "alpha"
        },
        "QuantileRegressor": {
            "loss_name": "quantile",
            "alpha_name": "quantile"
        },
        "HistGradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "alpha"
        },
        "LGBMRegressor": {
            "loss_name": "objective",
            "alpha_name": "alpha"
        },
}

def CQR_alphas(alphas, random_data, random_split, iters=0, funct=x_sinx, name_estimator="QuantileRegression"):
    mapie_coverage = []
    interval_width = []
    list_scores = []

    if iters>0:
        for i in range(iters):
            mc, iw, ls = CQR_alphas(alphas, random_data=random_data, random_split=random_split, funct=funct, name_estimator=name_estimator)
            mapie_coverage.append(mc)
            interval_width.append(iw)
            list_scores.append(ls)
    else:
        mapie_coverage_ = []
        interval_width_ = []
        list_scores_ = []
        random_state = np.random.randint(0, 1000)
        X_train_, X_calib_, y_train_, y_calib_, X_test, y_test, _ = get_data(funct, n_samples=1000, random_state=random_state, random_data=random_data, random_split=random_split, data_name="paper_reproduction")
        for alpha in alphas:
            estimator=get_estimator(name_estimator)
            mapie_reg = MapieQuantileRegressor(
                estimator=estimator,
                alpha=alpha,
            )
            mapie_reg.fit(X_train_, y_train_, X_calib_, y_calib_)
            y_pred, y_pis, = mapie_reg.predict(X_test, symmetry=symmetry)

            mapie_coverage_.append(regression_coverage_score(
                y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]
            ))
            list_scores_.append(mapie_reg.conformity_scores_)
            interval_width_.append(regression_mean_width_score(y_pis[:, 0, 0], y_pis[:, 1, 0]))
        return mapie_coverage_, interval_width_, list_scores_
    return mapie_coverage, interval_width, list_scores