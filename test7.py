import mlflow

runs = mlflow.search_runs(
    experiment_ids=["304250444131063630"],
    filter_string=f'tags."candidate" = "True"',
    output_format="list",
    order_by=["metrics.accuracy DESC"]
)
#print(runs)
for run in runs:
    print(run.info.run_name)
    print(run.data.metrics["accuracy"])
    print(run.data.params["model_config"])
    print("-"*30)