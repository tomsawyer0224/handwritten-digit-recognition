import logging
import click
import yaml
import os
from urllib.parse import urlparse

@click.group()
def run():
    pass

@click.command()
@click.option("--config_file", type=click.File("r"))
def init(config_file):
    os.makedirs("./scripts", exist_ok=True)
    # script to create virtual environment
    venv_cmds = [
        "virtualenv .venv",
        "source .venv/bin/activate",
        "pip install -U pip",
        "pip install -r requirements.txt"
    ]
    with open("./scripts/create_virtual_environment.sh", "w") as venv_scr:
        venv_scr.write("\n".join(venv_cmds))

    # script to start tracking server
    config = yaml.safe_load(config_file)
    tracking_uri = urlparse(config["mlflow"]["tracking_uri"])
    host_name = tracking_uri.hostname
    port = tracking_uri.port
    server_cmds = [
        "source .venv/bin/activate",
        f"mlflow server --host {host_name} --port {port}"
    ]
    with open("./scripts/start_tracking_server.sh", "w") as ts_scr:
        ts_scr.write("\n".join(server_cmds))

run.add_command(init)
if __name__ == "__main__":
    run()