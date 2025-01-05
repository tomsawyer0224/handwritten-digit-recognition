import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/a6af3dc4b1854c93bd60442f89016140/model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="handwritten-digit-recognition-model",
    enable_mlserver=True
)