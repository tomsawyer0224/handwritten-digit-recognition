import argparse
import logging
import mlflow
from mlflow import MlflowClient
from typing import Dict, Any
import more_itertools
import yaml
from core import (
    Trainer,
    Tuner,
    Digit_Data_Module
)

class HyperParamTuningPipeline:
    def __init__(
            self,
            model_configs: Dict[str, Dict[str, Any]],
            tuning_config: Dict[str, Any],
            data_module: Digit_Data_Module,
            mlflow_client: MlflowClient,
            experiment_id: str
        ) -> None:
        self.experiment_id = experiment_id
        self.data_module = data_module
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
        # tuning
        #more_itertools.consume((tuner.tune() for tuner in self.tuners))
        # find the best
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f'tags."candidate" = "good"',
            output_format="list",
            order_by=["metrics.accuracy DESC"]
        )
        best_run = runs[0]
        best_model_config = yaml.safe_load(best_run.data.params["model_config"])
        # re-train
        trainer = Trainer(
            model_config=best_model_config,
            data_module=self.data_module,
            experiment_id=self.experiment_id,
            run_name="best_model"
        )
        trainer.train()


    
