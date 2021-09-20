import numpy as np
import pandas as pd
import torch

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from models.config import *


def translate_column_shape(data: pd.DataFrame) -> pd.DataFrame:
    max_year = max([int(elem) for elem in set(data[COL_YEARS_RENEWED]) if elem != ELEM_NaN])
    new_col_elem = []

    for elem in data[COL_YEARS_RENEWED]:
        if elem == ELEM_NaN:
            new_col_elem.append('0')
        elif int(elem) <= (max_year % MAX_YEARS_RENEWED_LEVEL):
            new_col_elem.append('1')
        elif int(elem) <= (max_year % MAX_YEARS_RENEWED_LEVEL) * 2:
            new_col_elem.append('2')
        elif int(elem) <= (max_year % MAX_YEARS_RENEWED_LEVEL) * 3:
            new_col_elem.append('3')
        else:
            new_col_elem.append('4')

    data[COL_YEARS_RENEWED] = new_col_elem
    return data


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

    names = data.columns
    d = scale.fit_transform(data)
    normalized_data = pd.DataFrame(d, columns=names)

    return normalized_data


def prepare() -> [pd.DataFrame]:
    # work with path
    file_data_path = os.path.join(DATA_DIR, NAME_FILE)

    # read file with description and data
    data = pd.read_csv(file_data_path, sep=';', index_col=ID_COL)

    label_data_type = data[NAME_DATA_TYPE_COL].to_numpy()
    renewed = data[NAME_COL_RENEWED].to_numpy()

    del data[NAME_DATA_TYPE_COL]
    del data[NAME_COL_RENEWED]

    for name_col in DEL_COLUMN:
        del data[name_col]

    name_col_data = data.columns
    data = translate_column_shape(data)

    list_col_one_hot = [pd.get_dummies(data[name_col]).to_numpy()
                        for name_col in name_col_data if name_col in COLUMN_ONE_HOT]

    for name_col in COLUMN_ONE_HOT:
        del data[name_col]

    data = normalize_data(data)
    name_col_norm_data = data.columns
    data = data.to_numpy()
    width_tensor = max([el.shape[1] for el in list_col_one_hot])
    height_tensor = len(list_col_one_hot) + data.shape[1]
    len_train_sample = data.shape[0]

    train_sample = np.zeros((len_train_sample, height_tensor, width_tensor))

    for i in range(len_train_sample):
        for j in range(len(list_col_one_hot)):
            train_sample[i][j][:len(list_col_one_hot[j][i])] = list_col_one_hot[j][i]
        for j in range(data.shape[1]):
            train_sample[i][j + len(list_col_one_hot)][0] = data[i][j]

    train_x = np.asarray([train_sample[i] for i in range(len_train_sample) if label_data_type[i] == DATA_TYPE[0]],
                         dtype=np.float32)
    test = np.asarray([train_sample[i] for i in range(len_train_sample) if label_data_type[i] == DATA_TYPE[1]],
                      dtype=np.float32)
    train_y = np.asarray([renewed[i] for i in range(len_train_sample) if label_data_type[i] == DATA_TYPE[0]],
                         dtype=np.float32)

    train, val, train_targets, val_targets = train_test_split(train_x, train_y, random_state=42, train_size=0.9)

    train.to_csv(r'data\train.csv')
    train_targets.to_csv(r'data\train_targets.csv')
    val.to_csv(r'data\val.csv')
    val_targets.to_csv(r'data\val.csv')
    # add new dimension to axis=1
    train = np.expand_dims(train, axis=1)
    train_targets = np.expand_dims(train_targets, axis=1)
    val = np.expand_dims(val, axis=1)
    val_targets = np.expand_dims(val_targets, axis=1)

    return torch.from_numpy(train), torch.from_numpy(train_targets), \
           torch.from_numpy(val), torch.from_numpy(val_targets), test






