import mlflow
import numpy as np
import pandas as pd
from typing import Union, Dict, Any
from utils import create_model, id2name, name2id

#from core import Preprocessor, Digit_Data_Module

class Classifier(mlflow.pyfunc.PythonModel):
	def __init__(self, config: Dict[str, Any], use_default: bool = False) -> None:
		self.model = create_model(config=config, return_default_model=use_default)
		#self.preprocessor = preprocessor
		self.library = config.get("library")
	def fit(
			self,
			data: Union[np.ndarray, pd.DataFrame],
			target: Union[np.ndarray, pd.DataFrame],
			**kwargs
		) -> None:
		self.model.fit(data, target, **kwargs)
	def score(
			self,
			data: Union[np.ndarray, pd.DataFrame],
			target: Union[np.ndarray, pd.DataFrame]
		) -> float:
		return self.model.score(data, target)
	def load_context(self, context):
		pass
	def predict(self, context, model_input, params=None):
		#X = self.preprocessor(model_input)
		y_pred = self.model.predict(model_input)
		y_pred = id2name(y_pred)
		return y_pred
		
