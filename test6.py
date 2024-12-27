# Prerequisite: serve a custom pyfunc OpenAI model (not mlflow.openai) on localhost:5678
#   that defines inputs in the below format and params of `temperature` and `max_tokens`

import json
import requests
import numpy as np
import pandas as pd
from mlflow.models import convert_input_example_to_serving_input

#exam_path = "./mlartifacts/731750087336773565/adb5884fa5094860a20d0e6711db7859/artifacts/model/serving_input_example.json"
#exam_path = "./mlartifacts/731750087336773565/adb5884fa5094860a20d0e6711db7859/artifacts/model/input_example.json"
#exam_path = "./mlartifacts/731750087336773565/9216927d81fa45819e494b6c00d390f6/artifacts/xgboost_model/serving_input_example.json"
exam_path = "./mlartifacts/731750087336773565/1c770dde816849e5966aa88c2ab38bb0/artifacts/sklearn_model/serving_input_example.json"
with open(exam_path, "r") as f:
    payload_dict = json.load(f)
payload = json.dumps(payload_dict)
payload = convert_input_example_to_serving_input(
    pd.DataFrame(np.random.rand(4,28*28))
)
response = requests.post(
    #url=f"http://localhost:8080/invocations",
    url=f"http://localhost:5001/invocations",
    data=payload,
    headers={"Content-Type": "application/json"},
)
print(response.json())
