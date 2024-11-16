from sklearn.linear_model import (
    RidgeClassifier,
    LogisticRegression,
    SGDClassifier
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna


def get_model(config, default = False):
    """
    gets model from librares
    args:
        config: dict-like configuration
    return:
        classifier model
    """
    model_class = eval(config["model_class"])
    if config["model_params"].get("random_state") is None:
        config["model_params"]["random_state"] = 42
    if default:
        model = model_class(random_state = config["random_state"])
    else:
        model = model_class(**config["model_params"])
    return model

if __name__=="__main__":
    sklearn_config = dict(
        model_class = "RandomForestClassifier",
        model_params = dict(
            n_estimators = 50,
            max_depth = 10
        )
    )
    print(get_model(sklearn_config))
    xgboost_config = dict(
        model_class = "XGBClassifier",
        model_params = dict(
            tree_method="hist",
            early_stopping_rounds=3
        )
    )
    print(get_model(xgboost_config))
    lightgbm_config = dict(
        model_class = "LGBMClassifier",
        model_params = dict(
            boosting_type="gbdt",
            max_depth=3
        )
    )
    print(get_model(lightgbm_config))
    catboost_config = dict(
        model_class = "CatBoostClassifier",
        model_params = dict(
            iterations=500,
            depth=8
        )
    )
    print(get_model(catboost_config))
