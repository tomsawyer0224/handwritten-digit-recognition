import argparse
import logging
import mlflow
from mlflow import MlflowClient
from typing import Dict, Any
import more_itertools
from core import (
    Trainer,
    Tuner,
    Digit_Data_Module,
    Toy_Data_Module
)
from utils import load_config

class HyperParamTuningPipeline:
    def __init__(
            self,
            model_configs: Dict[str, Dict[str, Any]],
            tuning_config: Dict[str, Any],
            data_module: Digit_Data_Module,
            mlflow_client: MlflowClient,
            experiment_id: str
        ) -> None:
        #self.model_configs = model_configs
        self.tuners = [
            Tuner(
                model_config=model_config,
                tuning_config=tuning_config,
                data_module=data_module,
                mlflow_client=mlflow_client,
                experiment_id=experiment_id
            )
            for model_config in model_configs.values()
        ]
    def run_pipeline(self):
        more_itertools.consume((tuner.tune() for tuner in self.tuners))
    
