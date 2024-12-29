import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/c3b5fcfefa144b97815dc8ad3d28517c/model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="test_buil_docker",
    enable_mlserver=True
)