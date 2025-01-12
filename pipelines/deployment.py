import mlflow
import logging

# logging.basicConfig(
#         format="{asctime}::{levelname}::{name}::{message}",
#         style="{",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         level=logging.INFO
#     )
logger = logging.getLogger(__name__)
class DeploymentPipeline:
    def __init__(self, model_uri: str) -> None:
        self.model_uri = model_uri
    def run_pipeline(self):
        mlflow.models.build_docker(
            model_uri=self.model_uri,
            name="handwritten-digit-recognition-model",
            enable_mlserver=True
        )
        logger.info("the best model was deployed as a docker image named handwritten-digit-recognition-model")