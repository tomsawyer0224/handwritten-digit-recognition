import optuna
#from mlflow.tracking import MlflowClient
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

def get_objective(parent_run_id):
    # get an objective function for optuna that creates nested MLFlow runs

    def objective(trial):
        trial_run = client.create_run(
            experiment_id=experiment,
            tags={
                MLFLOW_PARENT_RUN_ID: parent_run_id
            }
        )

        x = trial.suggest_float("x", -10.0, 10.0)
        client.log_param(trial_run.info.run_id, "x", x)
        val =  x**2
        client.log_metric(trial_run.info.run_id, "xsq", val)
        return val
    
    return objective

client = MlflowClient()
experiment_name = "min_x_sq"
try:
    experiment = client.create_experiment(experiment_name)
except:
    experiment = client.get_experiment_by_name(experiment_name).experiment_id

study_run = client.create_run(experiment_id=experiment)
study_run_id = study_run.info.run_id

study = optuna.create_study(direction="minimize")
study.optimize(get_objective(study_run_id), n_trials=50, n_jobs=2)

client.log_param(study_run_id, "best_x", study.best_trial.params["x"])
client.log_metric(study_run_id, "best_xsq", study.best_value)