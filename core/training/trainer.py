from mlflow.models import infer_signature
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
            experiment_id: str,
            run_name: str = "best_model"
        ) -> None:
        self.model_config = model_config
        self.data_module = data_module
        self.experiment_id = experiment_id
        self.run_name = run_name
    def train(self):
        #preprocessor = self.data_module.get_preprocessor()
        #datasets = self.data_module.get_training_dataset()
        #inference_dataset = self.data_module.get_inference_dataset()
        #infer_data = inference_dataset["data"]
        #infer_target = inference_dataset["target"]
        #processed_infer_data = preprocessor(infer_data)
        #train_dataset = datasets["train_dataset"]
        #val_dataset = datasets["val_dataset"]
        train_dataset = self.data_module.train_dataset
        val_dataset = self.data_module.val_dataset
        test_dataset = self.data_module.test_dataset
        #clf = Classifier(config=self.model_config, preprocessor=preprocessor)
        clf = Classifier(config=self.model_config)
        signature = infer_signature(
                model_input=test_dataset["data"][:10],
                model_output=test_dataset["target"][:10]
            )
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attributes.run_name = '{self.run_name}'",
            output_format="list"
        )
        for run in runs:
            mlflow.delete_run(run_id=run.info.run_id)
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name):
            clf.fit(data=train_dataset["data"], target=train_dataset["target"])
            train_acc = clf.score(data=train_dataset["data"], target=train_dataset["target"])
            val_acc = clf.score(data=val_dataset["data"], target=val_dataset["target"])
            test_acc = clf.score(
                data=test_dataset["data"],
                target=test_dataset["target"]
            )
            mlflow.log_metrics(
                {"train_accuracy": train_acc, "val_accuracy": val_acc, "test_accuracy": test_acc}
            )
            mlflow.log_param("model_config", self.model_config)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=clf,
                signature=signature
            )