import sys
if "." not in sys.path: sys.path.append(".")

from mlflow import MlflowClient
import unittest
import logging

from core import Trainer, Digit_Data_Module
from utils import get_or_create_experiment

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
                n_estimators = 50,
                max_features = "sqrt"
            )
        )

xgboost_config = dict(
            library = "xgboost",
            model_class = "XGBClassifier",
            model_params = dict(
                tree_method="hist",
                early_stopping_rounds=3
            )
        )

lightgbm_config = dict(
            library = "lightgbm",
            model_class = "LGBMClassifier",
            model_params = dict(
                boosting_type="gbdt",
                max_depth=3
            )
        )

catboost_config = dict(
            library = "catboost",
            model_class = "CatBoostClassifier",
            model_params = dict(
                iterations=500,
                depth=8
            )
        )

logger.info("prepare digit data module")
data_module = Digit_Data_Module()

logger.info("create mlflow client")
mlflow_client = MlflowClient()

logger.info("create mlflow experiment")
experiment_name = "handwriten-digit-recognition"
experiment_id = get_or_create_experiment(experiment_name=experiment_name, client=mlflow_client)
class Test_Trainer(unittest.TestCase):
    """def test_Trainer_sklearn(self):
        print("training sklearn model")
        trainer = Trainer(
            model_config=sklearn_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="sklearn_best_model"
        )
        trainer.train()
        print("-"*30)"""
    """def test_Trainer_xgboost(self):
        print("training xgboost model")
        trainer = Trainer(
            model_config=xgboost_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="xgboost_best_model"
        )
        trainer.train()
        print("-"*30)"""
    def test_Trainer_lightgbm(self):
        print("training lightgbm model")
        trainer = Trainer(
            model_config=lightgbm_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="lightgbm_best_model"
        )
        trainer.train()
        print("-"*30)
    
    """def test_Trainer_catboost(self):
        print("training catboost model")
        trainer = Trainer(
            model_config=catboost_config,
            data_module=data_module,
            experiment_id=experiment_id,
            run_name="catboost_best_model"
        )
        trainer.train()
        print("-"*30)"""
if __name__=="__main__":
    unittest.main()