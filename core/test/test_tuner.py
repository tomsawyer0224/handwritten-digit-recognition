import sys
if "." not in sys.path: sys.path.append(".")

from mlflow import MlflowClient
import unittest
import logging

from core import Tuner, Digit_Data_Module, Toy_Data_Module
from utils import (
    get_or_create_experiment,
    
)

logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
logger = logging.getLogger(__name__)
sklearn_config = dict(
            library = "sklearn",
            model_class = "RandomForestClassifier",
            model_params = dict(
                n_jobs = 2,
                n_estimators = dict(
                    param_type = "int",
                    param_range = [50, 100]
                ),
                criterion = dict(
                    param_type = "categorical",
                    param_range = ["gini", "entropy", "log_loss"]
                ),
                max_depth = dict(
                    param_type = "int",
                    param_range = [2, 10],
                ),
                ccp_alpha = dict(
                    param_type = "float",
                    param_range = [0.0, 0.5]
                ),
                max_features = "sqrt",
                random_state = 42
            )
        )

xgboost_config = dict(
            library = "xgboost",
            model_class = "XGBClassifier",
            model_params = dict(
                n_estimators = dict(
                    param_type = "int",
                    param_range = [10,20]
                ),
                booster = dict(
                    param_type = "categorical",
                    param_range = ["gbtree", "gblinear"],

                ),
                learning_rate = dict(
                    param_type = "float",
                    param_range = [0.08,0.15]
                ),
                early_stopping_rounds = 5,
                verbosity=0,
                n_jobs = 2,
            )
        )

lightgbm_config = dict(
            library = "lightgbm",
            model_class = "LGBMClassifier",
            model_params = dict(
                max_depth=3,
                objective="multiclass",
                verbosity=-1,
                n_jobs = 2,
                n_estimators=dict(
                    param_type = "int",
                    param_range = [10,20]
                ),
                boosting_type=dict(
                    param_type = "categorical",
                    param_range = ["gbdt", "dart"],
                ),
                learning_rate=dict(
                    param_type = "float",
                    param_range = [0.08,0.15]
                )
            )
        )

catboost_config = dict(
            library = "catboost",
            model_class = "CatBoostClassifier",
            model_params = dict(
                iterations=5,
                loss_function="MultiClass",
                early_stopping_rounds=5,
                verbose=False,
                depth=8,
                learning_rate=dict(
                    param_type = "float",
                    param_range = [0.08,0.15]
                )
            )
        )

tuning_config = dict(
    n_trials = 3,
    n_jobs = 2
)

#logger.info("prepare digit data module")
#data_module = Digit_Data_Module()
logger.info("prepare toy data module")
data_module = Toy_Data_Module()

logger.info("create mlflow client")
mlflow_client = MlflowClient()

logger.info("create mlflow experiment")
experiment_name = "handwriten-digit-recognition"
experiment_id = get_or_create_experiment(experiment_name=experiment_name, client=mlflow_client)
class Test_Tuner(unittest.TestCase):
    """"""
    def test_Tuner_sklearn(self):
        tuner = Tuner(
            model_config=sklearn_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id
        )
        tuner.tune()
    
    def test_Tuner_xgboost(self):
        tuner = Tuner(
            model_config=xgboost_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id
        )
        tuner.tune()
    def test_Tuner_lightgbm(self):
        tuner = Tuner(
            model_config=lightgbm_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id
        )
        tuner.tune()
    def test_Tuner_catboost(self):
        tuner = Tuner(
            model_config=catboost_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id
        )
        tuner.tune()
if __name__=="__main__":
    unittest.main()