from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import pandas as pd

class Preprocessor:
    """
    a class for preprocessing data
    """
    def __init__(self) -> None:
        self.scaler = MinMaxScaler()
        #self.scaler = StandardScaler()
        #self.scaler = RobustScaler()
        self.fitted = False
    def __call__(self, data: np.ndarray|pd.DataFrame) -> np.ndarray|pd.DataFrame:
        if not self.fitted:
            return self.scaler.fit_transform(data)
        else:
            return self.scaler.transform(data)