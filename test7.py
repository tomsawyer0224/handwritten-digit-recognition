import mlflow
import json
import yaml

runs = mlflow.search_runs(
    experiment_ids=["614027524919096081"],
    filter_string=f'tags."candidate" = "good"',
    output_format="list",
    order_by=["metrics.accuracy DESC"]
)
#print(runs)
for run in runs:
    print(run.info.run_name)
    print(run.data.metrics["accuracy"])
    print(run.data.params["model_config"])
    print(type(run.data.params["model_config"]))
    model_config = yaml.safe_load(run.data.params["model_config"])
    print(model_config)
    print(type(model_config))
    print("-"*30)