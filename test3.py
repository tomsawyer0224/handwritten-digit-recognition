import optuna
import mlflow

def objective(trial):
    with mlflow.start_run(nested=True):                # Race condition possible
        x = trial.suggest_float("x", -10.0, 10.0)
        mlflow.log_param("x", x)                       # Race condition possible
        val = x**2
        mlflow.log_metric("xsq", val)                  # Race condition possible
        return val

with mlflow.start_run():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, n_jobs=2)