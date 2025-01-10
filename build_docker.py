import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
model_uri = "runs:/578413b18f414fdcb3328a452c30c326/model"
mlflow.models.build_docker(
    model_uri=model_uri,
    name="handwritten-digit-recognition-model",
    enable_mlserver=True,
    #base_image="python:{3.12}-slim"
)