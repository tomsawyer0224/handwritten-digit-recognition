import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/1c770dde816849e5966aa88c2ab38bb0/sklearn_model"
mlflow.models.build_docker(
    #model_uri="runs:/f1b7664ad57f40b6934d14f6cf47f8ec/model",
    #model_uri="runs:/9216927d81fa45819e494b6c00d390f6/xgboost_model",
    model_uri=model_uri,
    name="sklearn_test_buil_docker",
    enable_mlserver=True
)