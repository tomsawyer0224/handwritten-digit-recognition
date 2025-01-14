import mlflow
import logging
import yaml
import os

logger = logging.getLogger(__name__)
class DeploymentPipeline:
    def __init__(self, model_uri: str = None) -> None:
        if not os.path.isfile("./project_result.yaml"):
            raise Exception("It seems that the model has not been trained yet!")
        with open("./project_result.yaml", "r") as f:
            project_result = yaml.safe_load(f)
        #logger.info(f"{project_result=}")
        self.model_uri = model_uri if model_uri is not None else project_result["model_uri"]
        mlflow.set_tracking_uri(uri = project_result["tracking_uri"])
        #logger.info(f"{self.model_uri=}")

    def run_pipeline(self):
        try:
            mlflow.models.build_docker(
                model_uri=self.model_uri,
                name="handwritten-digit-recognition-model",
                enable_mlserver=True
            )
            logger.info("The best model was deployed as a docker image named handwritten-digit-recognition-model")
        except:
            print("Something went wrong with the model to be deployed!")