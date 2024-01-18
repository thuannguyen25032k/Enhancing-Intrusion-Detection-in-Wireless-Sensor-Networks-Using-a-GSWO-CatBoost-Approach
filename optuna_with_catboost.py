import numpy as np
from data_preprocessing import WSNDS_processing, KDD_processing, UNSW_NB15_processing, CICIDS2017_processing, WSNBFSF_processing, CICIDS2019_processing
import optuna
from optuna.integration import CatBoostPruningCallback

import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


def objective(trial: optuna.Trial) -> float:
    data_name = "KDD"
    if data_name == "KDD":
        print("Preprocessing for KDD")
        data_dir = "./data/NSL-KDD-Dataset"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = KDD_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
    elif data_name == "CICIDS2017":
        print("Preprocessing for CICIDS2017")
        data_dir = "./data/CICIDS17/MachineLearningCSV"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = CICIDS2017_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
    elif data_name == "CICIDS2019":
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = CICIDS2019_processing(data_dir)
    elif data_name == "WSNDS":
        print("Preprocessing for WSNDS")
        data_dir = "./data/WSN-DS"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = WSNDS_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
    elif data_name == "UNSW_NB15":
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = UNSW_NB15_processing(data_dir)
    elif data_name == "WSNBFSF":
        print("Preprocessing for WSNBFSF")
        data_dir = "./data/WSNBFSFDataset"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = WSNBFSF_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
    else:
        print("This type of data has not been processed yet")
    
    #seleced columns [ 0  3  4 10 12]
    sel_cols = [0, 1, 2, 4, 15, 21, 22, 23, 26, 27, 31, 32, 33, 34, 35, 36, 38, 40, 41]
    train_x = train_x.iloc[:, sel_cols]
    valid_x = valid_x.iloc[:, sel_cols]
    test_x = test_x.iloc[:, sel_cols]
    categorical_columns = list(train_x.select_dtypes(exclude=["number"]).columns)
    categorical_features_indices = train_x.columns.get_indexer(categorical_columns)

    param = {
        "objective": "MultiClass",
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 9),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "cat_features" : categorical_features_indices,
        "boosting_type": "Ordered",
        "bootstrap_type": "Bayesian",
        "used_ram_limit": "3gb",
        "eval_metric": "Accuracy",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = cb.CatBoostClassifier(**param)

    pruning_callback = CatBoostPruningCallback(trial, "Accuracy")
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=20,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(valid_y, pred_labels)

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))