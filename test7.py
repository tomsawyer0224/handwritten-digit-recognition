import mlflow

runs = mlflow.search_runs(
    experiment_ids=["304250444131063630"],
    filter_string=f'tags."candidate" = "True"',
    output_format="list"
)
#print(runs)
for run in runs:
    print(run.info.run_name)
    print(run)
    print("-"*30)