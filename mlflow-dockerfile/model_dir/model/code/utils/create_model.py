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

#import optuna

from typing import Dict, Any
#from .training_utilities import get_default_config, prepare_model_config

def create_model(model_config: Dict[str, Any]):
    """
    creates a model from configuration
    args:
        model_config: dict-like configuration
                e.g model_config = {
                    'library': 'sklearn',
                    'model_class': 'RandomForestClassifier',
                    'model_params': {
                        'n_estimators': 50,
                        'max_depth': 10
                    }
                }
    return:
        classifier model
    """
    model_class = eval(model_config["model_class"])
    model = model_class(**model_config["model_params"])
    return model
