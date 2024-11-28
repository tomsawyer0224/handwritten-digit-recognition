from mlflow import MlflowClient
import mlflow
#from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from typing import Dict, Any, Callable
from sklearn.utils import Bunch
#import logging

from core import Digit_Data_Module, Classifier

class Trainer:
    def __init__(
            self,
            model_config: Dict[str, Any],
            data_module: Digit_Data_Module,
            experiment_id: str
        ) -> None:
        self.model_config = model_config
        self.data_module = data_module
        self.experiment_id = self.experiment_id
    def train(self):
        preprocessor = self.data_module.get_preprocessor()
        datasets = self.data_module.get_training_dataset()
        inference_dataset = self.data_module.get_inference_dataset()
        train_dataset = datasets["train_dataset"]
        val_dataset = datasets["val_dataset"]
        clf = Classifier(config=self.model_config, preprocessor=preprocessor)
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attributes.run_name LIKE 'best_model'"
        )
        for run in runs:
            mlflow.delete_run(run_id=run.info.run_id)
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="best_model"):
            clf.fit(data=train_dataset["data"], target=train_dataset["target"])
            train_acc = clf.score(data=train_dataset["data"], target=train_dataset["target"])
            val_acc = clf.score(data=val_dataset["data"], target=val_dataset["target"])
            mlflow.log_metrics({"train_accuracy": train_acc, "val_accuracy": val_acc})
            mlflow.log_param("model_config", self.model_config)
