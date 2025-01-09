import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/2a85707d9c9a4314a90af800085eef16/model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="handwritten-digit-recognition-model",
    enable_mlserver=True,
    #base_image="python:{3.12}-slim"
)