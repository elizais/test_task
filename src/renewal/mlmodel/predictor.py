from catboost import CatBoostClassifier

from ..mlmodel.processor import get_data


class Predictor:
    def __init__(self, model_path):
        self._model = self._load_model(model_path)

    def __call__(self, file_path: str,
                 index_col: str,
                 name_data_type_col: str,
                 del_column: [str],
                 data_type: str):
        data_pred = self._preprocess(file_path, index_col, name_data_type_col, del_column, data_type)
        index_pred = list(data_pred.index)
        predict = self._model.predict(data_pred)
        predict = [{'id': index_pred[i], 'ren': predict[i]} for i in range(len(index_pred))]
        return predict

    def _load_model(self, model_path):
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model

    def _preprocess(self, file_path: str,
                    index_col: str,
                    name_data_type_col: str,
                    del_column: [str],
                    data_type: str):
        data = get_data(file_path, index_col, name_data_type_col, del_column, data_type)
        return data
