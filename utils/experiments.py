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