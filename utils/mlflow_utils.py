import mlflow
from mlflow import MlflowClient


def get_or_create_experiment(experiment_name: str, client: MlflowClient = None) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
        experiment_name (str): Name of the MLflow experiment.

    Returns:
        ID of the existing or newly created MLflow experiment.
    """
    if client is not None:
        if experiment := client.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return client.create_experiment(experiment_name)
    else:
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)
def get_next_run_name(experiment_id: str, prefix: str = "version") -> str:
    """
    create a new run name of a specific experiment (e.g version_0, version_1)
    Parameters:
        experiment_name: name of experiment that the run will be excuted
        prefix: prefix of run name
    Returns:
        run name
    """
    runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format = "list")
    run_names = [run.info.run_name for run in runs]
    if run_names:
        newest_run_ver = int(run_names[0].split("_")[-1])
    else:
        newest_run_ver = -1
    next_run_name = f"{prefix}_{newest_run_ver+1}"
    return next_run_name
