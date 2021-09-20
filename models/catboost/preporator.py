import numpy as np
import pandas as pd

from models.config import *


def get_data(file_data_path: str) -> [pd.DataFrame]:
    data = pd.read_csv(file_data_path, sep=';', index_col=ID_COL)

    renewal = data[NAME_COL_RENEWED]
    label = data[NAME_DATA_TYPE_COL].to_numpy()
    for name_col in DEL_COLUMN:
        del data[name_col]

    return data, label, renewal


def prepare_data(data: pd.DataFrame,
                 label: pd.DataFrame,
                 renew: np.asarray
                 ) -> [pd.DataFrame]:
    train = data.loc[label == DATA_TYPE[0]]
    target = renew.loc[label == DATA_TYPE[0]]

    return train, target
