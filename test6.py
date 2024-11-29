from mlflow.models import validate_serving_input
import logging

from core import Digit_Data_Module
model_uri = 'runs:/7921b9cfc90a4da896fe0d61c58c8d30/model'
logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
logger = logging.getLogger(__name__)
logger.info("prepare digit data module")
data_module = Digit_Data_Module()
inference_dataset = data_module.get_inference_dataset()
infer_data = inference_dataset["data"]
infer_target = inference_dataset["target"]
# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
serving_payload = convert_input_example_to_serving_input(infer_data)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
logger.info("Done validating the serving payload works on the model")

import mlflow
logged_model = 'runs:/7921b9cfc90a4da896fe0d61c58c8d30/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

loaded_model.predict(infer_data)
logger.info("Done predicting on inference data")