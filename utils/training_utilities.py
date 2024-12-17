import pandas as pd
import lightgbm as lbg
#from core import Classifier, Digit_Data_Module
from typing import Dict, Any, Union
import optuna

def id2name(label: pd.Series):
    return label.astype(str)
def name2id(label: pd.Series):
    return label.astype(int)
def get_fit_config(classifier, val_data, val_target):
    if classifier.library == "xgboost":
        fit_config = dict(
            eval_set=[(val_data, val_target)],
            verbose=False
        )
    elif classifier.library == "lightgbm":
        fit_config = dict(
            eval_set=[(val_data, val_target)],
            callbacks=[lbg.early_stopping(stopping_rounds=10, verbose=False)]
        )
    elif classifier.library == "catboost":
        fit_config = dict(
            eval_set=[(val_data, val_target)],
            verbose=False
        )
    else:
        fit_config = {}
    return fit_config
def prepare_training_data(train_dataset, val_dataset):
    train_data = train_dataset["data"]
    train_target = name2id(train_dataset["target"])
    val_data = val_dataset["data"]
    val_target = name2id(val_dataset["target"])
    return train_data, train_target, val_data, val_target
def prepare_model_config(
        model_config: Dict[str, Any],
        trial: optuna.trial.Trial = None,
        return_default_config: bool = False
    ) -> Dict[str, Any]:
    default_config = {
        k: v for k, v in model_config.items() if k != "model_params"
    }
    default_config["model_params"] = dict(
        random_state = model_config["model_params"].get("random_state", 42)
    )
    if return_default_config:
        return default_config
    final_config = {
        k: v for k, v in default_config.items()
    }
    if trial is None:
        final_config["model_params"] = default_config["model_params"]|model_config["model_params"]
        return final_config
    
    config = {k: v for k, v in model_config.items() if k != "model_params"}
    config["model_params"] = {}
    model_class = config["model_class"]
    for name, value in model_config["model_params"].items():
        if isinstance(value, dict):
            if value["param_type"] == "float":
                low, high = value["param_range"]
                config["model_params"][name] = trial.suggest_float(
                    name=f"{model_class}-{name}",
                    low=low,
                    high=high
                )
            elif value["param_type"] == "int":
                low, high = value["param_range"]
                config["model_params"][name] = trial.suggest_int(
                    name=f"{model_class}-{name}",
                    low=low,
                    high=high
                )
            elif value["param_type"] == "categorical":
                config["model_params"][name] = trial.suggest_categorical(
                    name=f"{model_class}-{name}",
                    choices=value["param_range"]
                )
            else:
                raise "param_type should be in ['float', 'int', 'categorical']"
        else:
            config["model_params"][name] = value
    final_config["model_params"] = default_config["model_params"]|config["model_params"]
    return final_config
def get_fixed_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    fixed_config = {
        k: v for k, v in model_config.items() if k != "model_params"
    }
    fixed_config["model_params"] = dict()
    for k, v in model_config["model_params"].items():
        if not isinstance(v, dict):
            fixed_config["model_params"][k] = v
    fixed_config["model_params"]["random_state"] = model_config["model_params"].get("random_state", 42)
    return fixed_config