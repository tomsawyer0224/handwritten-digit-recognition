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

    # create
    config = yaml.safe_load(config_file)
    tracking_uri = urlparse(config["mlflow"]["tracking_uri"])
    host_name = tracking_uri.hostname
    port = tracking_uri.port
    print(f"{host_name=}")
    print(f"{port=}")


run.add_command(init)
if __name__ == "__main__":
    run()