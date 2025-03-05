import mlflow
from model.code.core import Digit_Data_Module
import numpy as np

data_module = Digit_Data_Module()
test_dataset = data_module.test_dataset

loaded_model = mlflow.pyfunc.load_pyfunc(model_uri="model")
test_preds = loaded_model.predict(test_dataset["data"][:10])
print(f"predictions  = {test_preds}")