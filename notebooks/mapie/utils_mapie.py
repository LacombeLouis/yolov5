
import sys
sys.path.append('../../')
from utils.utils_numpy import get_binning_groups

import numpy as np
import pandas as pd
from matplotlib.offsetbox import TextArea, AnnotationBbox
import sklearn
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.datasets import (
    fetch_california_housing, load_diabetes, make_regression,
    make_sparse_uncorrelated, make_friedman1, make_friedman2, make_friedman3
)

from mapie.regression import MapieRegressor
from mapie.subsample import Subsample
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.metrics import (
    regression_coverage_score,
    regression_mean_width_score
)

def get_data(name, random_state, n_samples=10000, n_features=15):
    if name == "california":
        data = fetch_california_housing(as_frame=True)
        X = pd.DataFrame(data=data.data, columns=data.feature_names).drop(columns=['Latitude', 'Longitude'], axis=1)
        y = pd.Series(data=data.target)
    elif name == "diabetes":
        data = load_diabetes(as_frame=True)
        X = pd.DataFrame(data=data.data, columns=data.feature_names)
        y = pd.Series(data=data.target)
        X["sex1"] = 0
        X["sex2"] = 0
        X.loc[(X["sex"] == np.unique(X["sex"])[0]), "sex1"] = 1
        X.loc[(X["sex"] == np.unique(X["sex"])[1]), "sex2"] = 1
        X = X.drop(["sex"], axis=1)
    elif name == "make_regression":
        data = make_regression(n_samples, n_features, random_state=random_state)
        X = pd.DataFrame(data[0])
        y = pd.Series(data[1])
    elif name == "sparse_uncorrelated":
        data = make_sparse_uncorrelated(n_samples, n_features, random_state=random_state)
        X = pd.DataFrame(data[0])
        y = pd.Series(data[1])
    elif name == "friedman1":
        data = make_friedman1(n_samples, n_features, random_state=random_state)
        X = pd.DataFrame(data[0])
        y = pd.Series(data[1])
    elif name == "friedman2":
        data = make_friedman2(n_samples, random_state=random_state)
        X = pd.DataFrame(data[0])
        y = pd.Series(data[1])
    elif name == "friedman3":
        data = make_friedman3(n_samples, random_state=random_state)
        X = pd.DataFrame(data[0])
        y = pd.Series(data[1])
    elif name == "heteroscedastic":
        X = np.linspace(0, 5, n_samples)
        y = pd.Series((((3*X)+5) + (np.random.normal(0, 1, len(X)) * X)))
        X = pd.DataFrame(X)
    elif name == "homoscedastic":
        X = np.linspace(0, 5, n_samples)
        y = pd.Series((((3*X)+5) + (np.random.normal(0, 1, len(X)))))
        X = pd.DataFrame(X)
    return X, y

def sort_y_values(y_test, y_pred, y_pis):
    """
    Sorting the dataset in order to make plots using the fill_between function.
    """
    indices = np.argsort(y_test)
    y_test_sorted = np.array(y_test)[indices]
    y_pred_sorted = y_pred[indices]
    y_lower_bound = y_pis[:, 0, 0][indices]
    y_upper_bound = y_pis[:, 1, 0][indices]
    return y_test_sorted, y_pred_sorted, y_lower_bound, y_upper_bound


def plot_prediction_intervals(
    title,
    axs,
    y_test_sorted,
    y_pred_sorted,
    lower_bound,
    upper_bound,
    coverage,
    width,
    num_plots_idx,
    round_to=3
):
    """
    Plot of the prediction intervals for each different conformal
    method.
    """
    # axs.yaxis.set_major_formatter(FormatStrFormatter('%.0f' + "k"))
    # axs.xaxis.set_major_formatter(FormatStrFormatter('%.0f' + "k"))

    lower_bound_ = np.take(lower_bound, num_plots_idx)
    y_pred_sorted_ = np.take(y_pred_sorted, num_plots_idx)
    y_test_sorted_ = np.take(y_test_sorted, num_plots_idx)

    error = y_pred_sorted_-lower_bound_

    warning1 = y_test_sorted_ > y_pred_sorted_+error
    warning2 = y_test_sorted_ < y_pred_sorted_-error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=error[~warnings],
        capsize=5, marker="o", elinewidth=2, linewidth=0,
        label="Inside prediction interval"
        )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=error[warnings],
        capsize=5, marker="o", elinewidth=2, linewidth=0, color="red",
        label="Outside prediction interval"
        )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        marker="*", color="green",
        label="True value"
    )
    axs.set_xlabel("True y values")
    axs.set_ylabel("Predicted y values")
    median_lim = np.median([axs.get_xlim()])
    min_lim = np.min([axs.get_xlim(), axs.get_ylim()])
    max_lim = np.max([axs.get_xlim(), axs.get_ylim()])
    ab = AnnotationBbox(
        TextArea(
            f"Coverage: {np.round(coverage, round_to)}\n"
            + f"Interval width: {np.round(width, round_to)}"
        ),
        xy=(median_lim, max_lim),
        )
    lims = [
        min_lim,  # min of both axes
        max_lim,  # max of both axes
    ]
    axs.plot(lims, lims, '--', alpha=0.75, color="black", label="x=y")
    axs.add_artist(ab)
    axs.set_title(title, fontweight='bold')


def get_coverages_widths_by_bins(
    want,
    y_test,
    y_pred,
    lower_bound,
    upper_bound,
    STRATEGIES,
    bins,
    bins_strategy
):
    """
    Given the results from MAPIE, this function split the data
    according the the test values into bins and calculates coverage
    or width per bin.
    """
    # cuts = []
    # cuts_ = pd.qcut(y_test["naive"], bins).unique()[:-1]
    # for item in cuts_:
    #     cuts.append(item.left)
    # cuts.append(cuts_[-1].right)
    # cuts.append(np.max(y_test["naive"])+1)
    bins, _ = get_binning_groups(y_test["naive"], bins+1, bins_strategy)
    recap = {}
    for i in range(len(bins) - 1):
        cut1, cut2 = bins[i], bins[i+1]
        name = f"[{np.round(cut1, 2)}, {np.round(cut2, 2)}]"
        recap[name] = []
        for strategy in STRATEGIES:
            indices = np.where(
                (y_test[strategy] > cut1) * (y_test[strategy] <= cut2)
                )
            y_test_trunc = np.take(y_test[strategy], indices)
            y_low_ = np.take(lower_bound[strategy], indices)
            y_high_ = np.take(upper_bound[strategy], indices)
            if want == "coverage":
                recap[name].append(regression_coverage_score(
                    y_test_trunc[0],
                    y_low_[0],
                    y_high_[0]
                ))
            elif want == "width":
                recap[name].append(
                    regression_mean_width_score(y_low_[0], y_high_[0])
                )
    recap_df = pd.DataFrame(recap, index=STRATEGIES)
    return recap_df


def running_strategies(STRATEGIES, estimator, X_train, y_train, X_calib, y_calib, X_test, y_test, alpha):
    y_pred, y_pis = {}, {}
    y_test_sorted, y_pred_sorted, lower_bound, upper_bound = {}, {}, {}, {}
    coverage, width = {}, {}
    for strategy, params in STRATEGIES.items():
        if strategy == "cqr":
            mapie = MapieQuantileRegressor(estimator, **params)
            mapie.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
            y_pred[strategy], y_pis[strategy] = mapie.predict(X_test)
        else:
            mapie = MapieRegressor(estimator, **params)
            mapie.fit(X_train, y_train)
            y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=alpha)
        (
            y_test_sorted[strategy],
            y_pred_sorted[strategy],
            lower_bound[strategy],
            upper_bound[strategy]
        ) = sort_y_values(y_test, y_pred[strategy], y_pis[strategy])
        coverage[strategy] = regression_coverage_score(
            y_test,
            y_pis[strategy][:, 0, 0],
            y_pis[strategy][:, 1, 0]
            )
        width[strategy] = regression_mean_width_score(
            y_pis[strategy][:, 0, 0],
            y_pis[strategy][:, 1, 0]
            )
    return y_pred, y_pis, y_test_sorted, y_pred_sorted, lower_bound, upper_bound, coverage, width


def optimize_estimator(estimator, params_distributions, X_train, y_train):
    optim_model = RandomizedSearchCV(
            estimator,
            param_distributions=params_distributions,
            n_jobs=-1,
            n_iter=100,
            cv=KFold(n_splits=5, shuffle=True),
            verbose=-1
        )
    optim_model.fit(X_train, y_train)
    estimator = optim_model.best_estimator_
    return estimator
