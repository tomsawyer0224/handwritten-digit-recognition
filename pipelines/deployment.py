import mlflow
import logging
import yaml
# logging.basicConfig(
#         format="{asctime}::{levelname}::{name}::{message}",
#         style="{",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         level=logging.INFO
#     )
logger = logging.getLogger(__name__)
class DeploymentPipeline:
    def __init__(self, model_uri: str = None) -> None:
        with open("./project_result.yaml", "r") as f:
            project_result = yaml.safe_load(f)
        self.model_uri = model_uri if model_uri is not None else project_result["model_uri"]
        mlflow.set_tracking_uri = project_result["tracking_uri"]
    def run_pipeline(self):
        mlflow.models.build_docker(
            model_uri=self.model_uri,
            name="handwritten-digit-recognition-model",
            enable_mlserver=True
        )
        logger.info("the best model was deployed as a docker image named handwritten-digit-recognition-model")