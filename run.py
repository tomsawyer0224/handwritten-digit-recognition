import logging
import click
import yaml
import os
from urllib.parse import urlparse
import mlflow

from core import Digit_Data_Module, Toy_Data_Module
from pipelines import HyperParamTuningPipeline, DeploymentPipeline

logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

@click.group()
def run():
    pass

@click.command()
@click.option("-cf", "--config_file", type=click.File("r"), default="./config/project_config.yaml")
def prepare(config_file):
    os.makedirs("./scripts", exist_ok=True)
    
    # script to start tracking server
    config = yaml.safe_load(config_file)
    tracking_uri = config["mlflow"]["tracking_uri"]
    parsed_tracking_uri = urlparse(tracking_uri)
    host_name = parsed_tracking_uri.hostname
    port = parsed_tracking_uri.port
    server_start_cmds = [
        "source .venv/bin/activate",
        f"mlflow server --host {host_name} --port {port}"
    ]
    with open("./scripts/start_tracking_server.sh", "w") as ss_scr:
        ss_scr.write("\n".join(server_start_cmds))

    # script to stop tracking server
    with open("./scripts/stop_tracking_server.sh", "w") as sstp_scr:
        sstp_scr.write(
            "ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9"
        )
    
    # script to run docker
    with open("./scripts/run_docker.sh", "w") as rd_scr:
        rd_scr.write("docker run -p 5001:8080 handwritten-digit-recognition-model")
    
    # script to stop docker container
    with open("./scripts/stop_docker_container.sh", "w") as sdc_scr:
        sdc_scr.write(
            'docker ps --filter "ancestor=handwritten-digit-recognition-model" -q | xargs docker stop'
        )
    # script to set the tracking uri
    tu_cmds = [
        "source .venv/bin/activate",
        f"export MLFLOW_TRACKING_URI={tracking_uri}"
    ]
    with open("./scripts/set_tracking_uri.sh", "w") as stu_scr:
        stu_scr.write("\n".join(tu_cmds))
    click.echo("scripts are created in the 'scripts/' directory!")

@click.command()
@click.option("-cf", "--config_file", type=click.File("r"))
def tune(config_file):
    project_config = yaml.safe_load(config_file)
    data_module = Digit_Data_Module()
    # data_module = Toy_Data_Module()
    hp_tuning_ppl = HyperParamTuningPipeline(
            model_configs=project_config["models"],
            tuning_config=project_config["optuna"],
            data_module=data_module,
            tracking_uri=project_config["mlflow"]["tracking_uri"],
            experiment_name=project_config["mlflow"]["experiment_name"]
        )
    hp_tuning_ppl.run_pipeline()

@click.command()
@click.option("-m", "--model_uri", type=click.STRING, default=None)
def deploy(model_uri):
    deployment_ppl = DeploymentPipeline(model_uri=model_uri)
    deployment_ppl.run_pipeline()

run.add_command(prepare)
run.add_command(tune)
run.add_command(deploy)

if __name__ == "__main__":
    run()