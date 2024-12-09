import pandas as pd

def id2name(label: pd.Series):
    return label.astype(str)
def name2id(label: pd.Series):
    return label.astype(int)