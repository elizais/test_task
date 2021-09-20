import pandas as pd


def get_data(file_data_path: str,
             index_col: str,
             name_data_type_col: str,
             del_column: [str],
             data_type: str) -> [pd.DataFrame]:
    data = pd.read_csv(file_data_path, sep=';', index_col=index_col)

    label = data[name_data_type_col].to_numpy()
    for name_col in del_column:
        del data[name_col]

    test = data.loc[label == data_type]
    return test
