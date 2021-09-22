import os

class BaseConfig:
    REN_DIR = 'renewal'
    FINAL_MODEL_DIR = 'mlmodel'
    NAME_FINAL_MODEL = 'renewal_model.cbm'
    NAME_FINAL_MODEL_DIR = os.path.join(REN_DIR, FINAL_MODEL_DIR, NAME_FINAL_MODEL)

    NAME_DATA_TYPE_COL = 'DATA_TYPE'
    ID_COL = 'POLICY_ID'
    NAME_COL_RENEWED = 'POLICY_IS_RENEWED'

    DEL_COLUMN = [
        'DATA_TYPE',
        'POLICY_IS_RENEWED',
        'POLICY_SALES_CHANNEL',
        'VEHICLE_MAKE',
        'VEHICLE_MODEL',
        'POLICY_INTERMEDIARY',
        'POLICY_YEARS_RENEWED_N',
        'CLIENT_REGISTRATION_REGION',
    ]

    DATA_TYPE = ['TRAIN', 'TEST ']

    UPLOAD_FOLDER = ''

    ALLOWED_EXTENSIONS = {'txt', 'csv'}


class DevConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False

