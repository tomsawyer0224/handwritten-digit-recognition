import mlflow
import numpy as np
import pandas as pd
from typing import Union, Dict, Any
from utils import create_model, id2name

class Classifier(mlflow.pyfunc.PythonModel):
#class Classifier:
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
	# def load_context(self, context):
	# 	pass
	def predict(self, context, model_input, params=None):
		y_pred = self.model.predict(model_input)
		y_pred = id2name(y_pred)
		return y_pred
	def get_prediction(self, data: Union[np.ndarray, pd.DataFrame]):
		prediction = self.model.predict(data)
		prediction = id2name(prediction)
		return prediction
# class MLflowModel(mlflow.pyfunc.PythonModel):
# 	def __init__(self, classifier):
# 		self.classifier = classifier
# 	def predict(self, context, model_input, params=None):
# 		# y_pred = self.classifier.model.predict(model_input)
# 		y_pred = id2name(y_pred)
# 		y_pred = self.classifier.get_prediction(model_input)
# 		# y_pred = self.classifier.predict(model_input)
# 		# y_pred = id2name(y_pred)
# 		return y_pred
		
