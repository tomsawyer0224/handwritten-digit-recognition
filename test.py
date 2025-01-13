import yaml

info = dict(model_uri = "runs:/abcd")
with open("info.yaml", "w") as f:
    yaml.dump(info, f)

with open("info.yaml", "r") as f:
    conf = yaml.safe_load(f)
print(conf)
