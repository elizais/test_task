from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from preporator import *
from models.config import *

np.set_printoptions(precision=4)


if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, NAME_FILE)
    dataset, label_data_type, renewed = get_data(file_path)
    cat_features = np.where(dataset.dtypes != float)[0]
    features, targets = prepare_data(dataset, label_data_type, renewed)

    test = dataset.loc[label_data_type == DATA_TYPE[1]]
    features, val, targets, val_targets = train_test_split(features, targets, random_state=42, train_size=0.85)

    model = CatBoostClassifier(
        random_seed=63,
        iterations=10000,
        learning_rate=0.01,
        loss_function='Logloss',
        boosting_type='Ordered',
        leaf_estimation_method='Newton',
        custom_loss=['AUC', 'Accuracy'],
        early_stopping_rounds=100,
        # properties for training a model on a GPU
        # task_type="GPU",
        # devices='0:1'
    )

    model.fit(
        features, targets,
        cat_features=cat_features,
        verbose=False,
        eval_set=(val, val_targets),
        plot=True
    )

    filename = 'catboost_model_train_customer.cbm'
    model.save_model(os.path.join(CHECKPOINT_DIR, filename))
