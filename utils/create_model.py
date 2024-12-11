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

from typing import Dict, Any

def create_model(config: Dict[str, Any], return_default_model: bool = False):
    """
    creates model from config
    args:
        config: dict-like configuration
                e.g config = {
                    'library': 'sklearn',
                    'model_class': 'RandomForestClassifier',
                    'model_params': {
                        'n_estimators': 50,
                        'max_depth': 10
                    }
                }
        return_default_model: return the default model (ignores the 'config' param)
    return:
        classifier model
    """
    model_class = eval(config["model_class"])
    if config["model_params"].get("random_state") is None:
        config["model_params"]["random_state"] = 42

    if config["model_params"].get("early_stopping_rounds") is None:
        if config["library"] in ["xgboost", "catboost"]:
            config["model_params"]["early_stopping_rounds"] = 10
    
    if return_default_model:
        default_config = dict(
            random_state = config["model_params"]["random_state"],
        )
        if default_config.get("early_stopping_rounds") is not None:
            default_config["early_stopping_rounds"] = config["model_params"]["early_stopping_rounds"]
        model = model_class(**default_config)
    else:
        model = model_class(**config["model_params"])
    return model
