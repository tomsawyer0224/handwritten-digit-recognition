import mlflow
import numpy as np
import pandas as pd
from typing import Union, Dict, Any
from utils import create_model, id2name

class Classifier(mlflow.pyfunc.PythonModel):
	def __init__(self, model_config: Dict[str, Any]) -> None:
		self.model = create_model(model_config=model_config)
		self.library = model_config.get("library")
	def fit(
			self,
			data: Union[np.ndarray, pd.DataFrame],
			target: Union[np.ndarray, pd.Series],
			**kwargs
		) -> None:
		self.model.fit(data, target, **kwargs)
	def score(
			self,
			data: Union[np.ndarray, pd.DataFrame],
			target: Union[np.ndarray, pd.Series]
		) -> float:
		return self.model.score(data, target)
	def predict(self, context, model_input, params=None):
		prediction = self.get_prediction(model_input)
		return prediction
	def get_prediction(self, data: Union[np.ndarray, pd.DataFrame]):
		prediction = self.model.predict(data)
		if prediction.ndim == 2:
			prediction = prediction.squeeze()
		return id2name(prediction)
		
