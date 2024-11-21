import sys
if "." not in sys.path: sys.path.append(".")

from mlflow import MlflowClient
import unittest
import logging

from core import Tuner, Digit_Data_Module
from utils import get_or_create_experiment

logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
logger = logging.getLogger(__name__)
sklearn_config = dict(
            model_class = "RandomForestClassifier",
            model_params = dict(
                n_estimators = 50,
                criterion = dict(
                    param_type = "categorical",
                    param_range = ["gini", "entropy", "log_loss"]
                ),
                max_depth = 10
            )
        )

xgboost_config = dict(
            model_class = "XGBClassifier",
            model_params = dict(
                tree_method="hist",
                early_stopping_rounds=3
            )
        )

lightgbm_config = dict(
            model_class = "LGBMClassifier",
            model_params = dict(
                boosting_type="gbdt",
                max_depth=3
            )
        )

catboost_config = dict(
            model_class = "CatBoostClassifier",
            model_params = dict(
                iterations=500,
                depth=8
            )
        )

tuning_config = dict(
    n_trials = 5,
    n_jobs = -1
)

logger.info("prepare digit data module")
data_module = Digit_Data_Module()

logger.info("create mlflow client")
mlflow_client = MlflowClient()

logger.info("create mlflow experiment")
experiment_name = "handwriten-digit-recognition"
experiment_id = get_or_create_experiment(experiment_name=experiment_name, client=mlflow_client)
class Test_Tuner(unittest.TestCase):
    def test_Tuner_sklearn(self):
        tuner = Tuner(
            model_config=sklearn_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id
        )
        tuner.tune()

if __name__=="__main__":
    unittest.main()