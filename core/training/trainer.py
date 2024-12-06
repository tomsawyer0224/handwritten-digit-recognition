from mlflow.models import infer_signature
import mlflow
#from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from typing import Dict, Any, Callable
from sklearn.utils import Bunch
#import logging
import numpy as np

from core import Digit_Data_Module, Classifier
from utils import visualize_image, visualize_confusion_matrix, visualize_classification_report

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
        #test_dataset = self.data_module.test_dataset
        #clf = Classifier(config=self.model_config, preprocessor=preprocessor)
        clf = Classifier(config=self.model_config)
        signature = infer_signature(
                model_input=val_dataset["data"][:2],
                model_output=val_dataset["target"][:2]
            )
        input_example = val_dataset["data"][:2]
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attributes.run_name = '{self.run_name}'",
            output_format="list"
        )
        for run in runs:
            mlflow.delete_run(run_id=run.info.run_id)
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name):
            # train
            train_image = visualize_image(
                dataset=train_dataset,
                prediction=None,
                nrows=4,
                ncols=4,
                figsize=(10,10),
                name="training dataset"
            )
            mlflow.log_figure(
                figure=train_image,
                artifact_file="report/training_images.png"
            )
            if clf.library == "xgboost":
                fit_config = dict(eval_set=[(val_dataset["data"], val_dataset["target"])])
            else:
                fit_config = {}
            clf.fit(data=train_dataset["data"], target=train_dataset["target"], **fit_config)
            train_acc = clf.score(data=train_dataset["data"], target=train_dataset["target"])
        
            # validate
            val_acc = clf.score(data=val_dataset["data"], target=val_dataset["target"])
            mlflow.log_metrics(
                {"train_accuracy": train_acc, "val_accuracy": val_acc}
            )
            val_predictions = clf.model.predict(val_dataset["data"])

            val_image = visualize_image(
                dataset=val_dataset,
                prediction=val_predictions,
                nrows=4,
                ncols=4,
                figsize=(10,10),
                name="validation dataset"
            )
            mlflow.log_figure(
                figure=val_image,
                artifact_file="report/val_images.png"
            )

            # log confusion matrix
            cm = visualize_confusion_matrix(
                y_true=val_dataset["target"],
                y_pred=val_predictions,
                name="confusion matrix"
            )
            mlflow.log_figure(
                figure=cm,
                artifact_file="report/confusion_matrix.png"
            )

            # log classification report
            cls_rpt = visualize_classification_report(
                y_true=val_dataset["target"],
                y_pred=val_predictions
            )
            mlflow.log_text(
                text=cls_rpt,
                artifact_file="report/classification_report.txt"
            )
            # log parameters
            mlflow.log_param("model_config", self.model_config)
            
            # log model
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=clf,
                signature=signature,
                input_example=input_example
            )