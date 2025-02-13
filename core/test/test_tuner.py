import sys

if "." not in sys.path:
    sys.path.append(".")

from mlflow import MlflowClient
import unittest
import logging

from core import Tuner, Digit_Data_Module, Toy_Data_Module
from utils import get_or_create_experiment, load_config

logging.basicConfig(
    format="{asctime}::{levelname}::{name}::{message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
sklearn_config = load_config("config/test/sklearn_tuning_config.yaml")
xgboost_config = load_config("config/test/xgboost_tuning_config.yaml")
lightgbm_config = load_config("config/test/lightgbm_tuning_config.yaml")
catboost_config = load_config("config/test/catboost_tuning_config.yaml")
tuning_config = load_config("config/test/tuning_config.yaml")
# logger.info("prepare digit data module")
# data_module = Digit_Data_Module()
logger.info("prepare toy data module")
data_module = Toy_Data_Module()

logger.info("create mlflow client")
mlflow_client = MlflowClient()

logger.info("create mlflow experiment")
experiment_name = "handwriten-digit-recognition"
experiment_id = get_or_create_experiment(
    experiment_name=experiment_name, client=mlflow_client
)


class Test_Tuner(unittest.TestCase):

    def test_Tuner_sklearn(self):
        tuner = Tuner(
            model_config=sklearn_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id,
        )
        tuner.tune()

    def test_Tuner_xgboost(self):
        tuner = Tuner(
            model_config=xgboost_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id,
        )
        tuner.tune()

    def test_Tuner_lightgbm(self):
        tuner = Tuner(
            model_config=lightgbm_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id,
        )
        tuner.tune()

    def test_Tuner_catboost(self):
        tuner = Tuner(
            model_config=catboost_config,
            tuning_config=tuning_config,
            data_module=data_module,
            mlflow_client=mlflow_client,
            experiment_id=experiment_id,
        )
        tuner.tune()


if __name__ == "__main__":
    unittest.main()
