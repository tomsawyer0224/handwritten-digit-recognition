import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/cab9af4cc7664be9b7c9740ee68efe48/model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="handwritten-digit-recognition-model",
    enable_mlserver=True,
    #base_image="python:{3.12}-slim"
)