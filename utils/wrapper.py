import mlflow

class Wrapper(mlflow.pyfunc.PythonModel):
	def __init__(self, config):
		pass
	def load_context(self, context):
		pass
	def predict(self, context, model_input, params=None):
		pass
		
