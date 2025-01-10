import argparse
import logging
import mlflow
from mlflow import MlflowClient
import click
import yaml

@click.group()
def cli():
    pass

#@cli.command()
@click.command()
@click.argument("a", type=click.FLOAT)
@click.argument("b", type=click.FLOAT)
def add(a, b):
    click.echo(a + b)

cli.add_command(add)
if __name__ == "__main__":
    cli()