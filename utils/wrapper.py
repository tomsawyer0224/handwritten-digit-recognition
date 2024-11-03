import mlflow
import numpy as np
import pandas as pd
from typing import Union
from get_model import get_model

class Classifier(mlflow.pyfunc.PythonModel):
	def __init__(self, model_config, preprocessor):
		self.model = get_model(model_config)
		self.preprocessor = preprocessor
	def fit(
			self,
			data: Union[np.ndarray, pd.DataFrame],
			target: Union[np.ndarray, pd.DataFrame]
		) -> None:
		self.model.fit(data, target)
	def score(
			self,
			data: Union[np.ndarray, pd.DataFrame],
			target: Union[np.ndarray, pd.DataFrame]
		) -> None:
		return self.model.score(data, target)
	def load_context(self, context):
		pass
	def predict(self, context, model_input, params=None):
		X = self.preprocessor(model_input)
		y_pred = self.model(X)
		return y_pred
		
