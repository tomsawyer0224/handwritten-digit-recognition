import numpy as np
import pandas as pd
from typing import Union, Dict, Any

from utils import create_model
class BaseModel:
    def __init__(self, model_config: Dict[str, Any], return_default_model: bool = False):
        self.library = model_config.get("library")
        self.name = model_config.get("model_class")
        self.model = create_model(config=model_config, return_default_model=return_default_model)
    def fit(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> None:
        self.model.fit(data, target, **kwargs)
    def score(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.DataFrame]
    ) -> float:
        return self.model.score(data, target)
    def predict(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]: 
        return self.model.predict(data, target)
    def save(self, path: str):
        pass