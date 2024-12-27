import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import SGDClassifier
import logging

from core import Digit_Data_Module
from utils import get_or_create_experiment, name2id

logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
logger = logging.getLogger(__name__)
logger.info("prepare digit data module")

data_module = Digit_Data_Module()

clf = SGDClassifier(random_state=42)

train_dataset = data_module.train_dataset
train_data = train_dataset["data"]
train_target = train_dataset["target"]

tracking_uri = "http://127.0.0.1:8000"
experiment_name = "handwriten-digit-recognition"
mlflow_client = MlflowClient(tracking_uri=tracking_uri)
mlflow.set_tracking_uri(uri=tracking_uri)

signature = infer_signature(
    model_input=train_data[:2],
    model_output=train_target[:2]
)
input_example = train_data[:2]
experiment_id = get_or_create_experiment(experiment_name=experiment_name, client=mlflow_client)
with mlflow.start_run(
    experiment_id=experiment_id,
    tags={"type": "sklearn_test"},
):
    clf.fit(train_data, name2id(train_target))
    mlflow.sklearn.log_model(
        clf,
        artifact_path="sklearn_model",
        signature=signature,
        input_example=input_example,
        #registered_model_name="handwritten-digit-recognition-model"
    )