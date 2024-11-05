import tempfile
import os
import mlflow
from mlflow import MlflowClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
from utils.mlflow_utils import get_or_create_experiment, get_next_run_name

X_train = np.random.randint(0,255,(10,36))
y_train = np.random.randint(0,10,(10,))

X_test = np.random.randint(0,255,(4,36))
y_test = np.random.randint(0,10,(4,))
preprocessor = MinMaxScaler()
model = LogisticRegression()

X_train = preprocessor.fit_transform(X_train)
model.fit(X_train, y_train)

tmp_dir = tempfile.TemporaryDirectory()
tmp_prep_path = os.path.join(tmp_dir.name, "preprocessor.joblib")
joblib.dump(preprocessor, tmp_prep_path)
tmp_model_path = os.path.join(tmp_dir.name, "fitted_model.joblib")
joblib.dump(model, tmp_model_path)

#experiment_id = get_or_create_experiment("Test PythonModel")
client = MlflowClient()
experiment_id = get_or_create_experiment("Test Nested Run", client)
artifacts = {"preprocessor": tmp_prep_path, "model": tmp_model_path}

class Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.preprocessor = None
        self.model = None
    def load_context(self, context):
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        self.model = joblib.load(context.artifacts["model"])
    def predict(self, context, model_input, params = None):
        X = self.preprocessor.transform(model_input)
        pred = self.model.predict(X)
        return pred

run_name = get_next_run_name(experiment_id=experiment_id)
with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="parent_run",
        nested=True
    ) as parent_run:
    mlflow.log_params(dict(parent_param_1=1, parent_param_2=2))
    with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="child_run_1",
            parent_run_id=parent_run.info.run_id,
            nested=True
        ) as child_run_1:
        mlflow.log_params(dict(child_1_param_1 = 11, child_1_param_2=12))
    with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="child_run_2",
            parent_run_id=parent_run.info.run_id,
            nested=True
        ) as child_run_2:
        mlflow.log_params(dict(child_2_param_1 = 21, child_2_param_2=22))
"""
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    print(run.info.run_name)
    mlflow.pyfunc.log_model(
        artifact_path = "classifier_model",
        python_model = Wrapper(),
        artifacts = artifacts
    )
"""
