import yaml


def load_config(config_file: str):
    with open(config_file, "r") as cf:
        config = yaml.safe_load(cf)
    return config
