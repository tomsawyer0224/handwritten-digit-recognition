import mlflow.artifacts
import mlflow.artifacts
from mlflow.models import infer_signature, validate_serving_input
import mlflow
from typing import Dict, Any
import json
import numpy as np

from core import Digit_Data_Module, Classifier

from utils import (
    visualize_image,
    visualize_confusion_matrix,
    visualize_classification_report,
    prepare_training_data,
    get_fit_config,
    prepare_model_config,
    id2name,
)


class Trainer:
    def __init__(
        self,
        model_config: Dict[str, Any],
        data_module: Digit_Data_Module,
        experiment_id: str,
        run_name: str = "best_model",
    ) -> None:
        self.model_config = model_config
        self.data_module = data_module
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.run_id = None

    @property
    def model_uri(self):
        return f"runs:/{self.run_id}/model"

    @property
    def artifact_uri(self):
        return f"mlflow-artifacts:/{self.experiment_id}/{self.run_id}/artifacts/model"

    def train(self):
        train_dataset = self.data_module.train_dataset
        val_dataset = self.data_module.val_dataset
        model_config = prepare_model_config(
            model_config=self.model_config, trial=None, return_default_config=False
        )
        clf = Classifier(model_config=model_config)
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attributes.run_name = '{self.run_name}'",
            output_format="list",
        )
        for run in runs:
            mlflow.delete_run(run_id=run.info.run_id)
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name,
            tags={"candidate": "best"},
        ) as best_run:
            # visualize training images
            train_image = visualize_image(
                dataset=train_dataset,
                prediction=None,
                nrows=4,
                ncols=4,
                figsize=(10, 10),
                name="training dataset",
            )
            mlflow.log_figure(
                figure=train_image, artifact_file="report/train_images.png"
            )

            # prepare for training
            train_data, train_target, val_data, val_target = prepare_training_data(
                train_dataset=train_dataset, val_dataset=val_dataset
            )
            fit_config = get_fit_config(
                classifier=clf, val_data=val_data, val_target=val_target
            )

            # train the model
            clf.fit(data=train_data, target=train_target, **fit_config)

            # log the training accuracy
            train_acc = clf.score(data=train_data, target=train_target)
            mlflow.log_metric(key="train_accuracy", value=train_acc)

            # log the validation accuracy
            val_acc = clf.score(data=val_data, target=val_target)
            mlflow.log_metric(key="val_accuracy", value=val_acc)

            # predict on the validation dataset
            val_predictions = clf.get_prediction(data=val_data)

            # visualize predictions on the validation dataset
            val_image = visualize_image(
                dataset=val_dataset,
                prediction=val_predictions,
                nrows=4,
                ncols=4,
                figsize=(10, 10),
                name="validation dataset",
            )
            mlflow.log_figure(figure=val_image, artifact_file="report/val_images.png")

            # confusion matrix
            cm = visualize_confusion_matrix(
                y_true=id2name(val_dataset["target"]),
                y_pred=val_predictions,
                name="confusion matrix",
            )
            mlflow.log_figure(figure=cm, artifact_file="report/confusion_matrix.png")

            # classification report
            cls_rpt = visualize_classification_report(
                y_true=id2name(val_dataset["target"]), y_pred=val_predictions
            )
            mlflow.log_text(
                text=cls_rpt, artifact_file="report/classification_report.txt"
            )

            # log parameters
            mlflow.log_param("model_config", model_config)

            # log model
            input_example = val_dataset["data"][:2]
            signature = infer_signature(
                model_input=input_example,
                model_output=clf.get_prediction(input_example),
            )
            # model_info = mlflow.pyfunc.log_model(
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=clf,
                signature=signature,
                input_example=input_example,
                # infer_code_paths=True,
                code_paths=["core", "utils"],
                pip_requirements="./requirements.txt",
            )
        self.run_id = best_run.info.run_id

    def test(self):
        print("validate serving input")
        serving_input_example = mlflow.artifacts.load_dict(
            f"{self.artifact_uri}/serving_input_example.json"
        )
        serving_payload = json.dumps(serving_input_example)
        val_preds = validate_serving_input(self.model_uri, serving_payload)
        print(f"val_preds = {val_preds}")
        print("-" * 30)
        print("inference")
        test_dataset = self.data_module.test_dataset
        loaded_model = mlflow.pyfunc.load_model(self.model_uri)
        test_preds = loaded_model.predict(test_dataset["data"][:10])
        print(f"predictions  = {test_preds}")
        ground_truth = np.array(id2name(test_dataset["target"][:10]))
        print(f"ground truth = {ground_truth}")
