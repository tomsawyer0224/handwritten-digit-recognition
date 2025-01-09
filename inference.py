import json
import requests
import numpy as np
import pandas as pd
from mlflow.models import convert_input_example_to_serving_input
from core import Digit_Data_Module

data_module = Digit_Data_Module()
test_dataset = data_module.test_dataset
inference_payload = convert_input_example_to_serving_input(test_dataset["data"][:50])
response = requests.post(
    url=f"http://127.0.0.1:5001/invocations",
    data=inference_payload,
    headers={"Content-Type": "application/json"},
)
print(response.json())
