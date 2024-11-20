from mlflow import MlflowClient

def get_or_create_experiment(experiment_name: str, client: MlflowClient) -> str:
    """
    gets an existing experiment or creates a new experiment
    args:
        experiment_name: the name of experiment
        client: an instance of MlflowClient
    returns:
        experiment ID
    """
    if experiment := client.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return client.create_experiment(experiment_name)
    
def generate_next_run_name(
        mlflow_client: MlflowClient,
        experiment_id: str,
        prefix: str = "version"
    ) -> str:
    runs = mlflow_client.search_runs(experiment_ids=[experiment_id])
    run_names = [run.info.run_name for run in runs]
    if run_names:
        newest_run_ver = int(run_names[0].split("_")[-1])
    else:
        newest_run_ver = -1
    next_run_name = f"{prefix}_{newest_run_ver+1}"
    return next_run_name