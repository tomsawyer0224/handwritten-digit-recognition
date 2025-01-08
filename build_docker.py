import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/495a78fc368843e78cf5152c6f135c5e/model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="handwritten-digit-recognition-model",
    enable_mlserver=True,
    base_image="python:{3.12}-slim"
)