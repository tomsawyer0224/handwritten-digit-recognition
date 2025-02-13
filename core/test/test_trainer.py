import sys

if "." not in sys.path:
    sys.path.append(".")

from mlflow import MlflowClient
import unittest
import logging

from core import Trainer, Digit_Data_Module, Toy_Data_Module
from utils import get_or_create_experiment, load_config

logging.basicConfig(
    format="{asctime}::{levelname}::{name}::{message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
sklearn_config = load_config("config/test/sklearn_training_config.yaml")
xgboost_config = load_config("config/test/xgboost_training_config.yaml")
lightgbm_config = load_config("config/test/lightgbm_training_config.yaml")
catboost_config = load_config("config/test/catboost_training_config.yaml")

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


class Test_Trainer(unittest.TestCase):

    def test_Trainer_sklearn(self):
        print("training sklearn model")
        trainer = Trainer(
            model_config=sklearn_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="sklearn_best_model",
        )
        trainer.train()
        print("-" * 30)

    def test_Trainer_xgboost(self):
        print("training xgboost model")
        trainer = Trainer(
            model_config=xgboost_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="xgboost_best_model",
        )
        trainer.train()
        print("-" * 30)

    def test_Trainer_lightgbm(self):
        print("training lightgbm model")
        trainer = Trainer(
            model_config=lightgbm_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="lightgbm_best_model",
        )
        trainer.train()
        print("-" * 30)

    def test_Trainer_catboost(self):
        print("training catboost model")
        trainer = Trainer(
            model_config=catboost_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="catboost_best_model",
        )
        trainer.train()
        print("-" * 30)


if __name__ == "__main__":
    unittest.main()
