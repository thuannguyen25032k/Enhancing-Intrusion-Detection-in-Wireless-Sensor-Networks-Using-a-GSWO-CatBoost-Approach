import numpy as np
import pandas as pd
from data_preprocessing import WSNDS_processing, KDD_processing, UNSW_NB15_processing, CICIDS2017_processing, WSNBFSF_processing, CICIDS2019_processing
from catboost import CatBoostClassifier, Pool, metrics, cv
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# Define the objective function for Bayesian Optimization
def catboost_cv(depth, learning_rate, iterations, random_strength, l2_leaf_reg):
    # Convert hyperparameters to the right format
    data_name = "CICIDS2017"
    if data_name == "KDD":
        print("Preprocessing for KDD")
        data_dir = "./data/NSL-KDD-Dataset"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = KDD_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
        sel_cols = [0, 1, 2, 4, 15, 21, 22, 23, 26, 27, 31, 32, 33, 34, 35, 36, 38, 40, 41]
        X_train = train_x.iloc[:, sel_cols]
        X_valid = valid_x.iloc[:, sel_cols]
        X_test = test_x.iloc[:, sel_cols]
    elif data_name == "CICIDS2017":
        print("Preprocessing for CICIDS2017")
        data_dir = "./data/CICIDS17/MachineLearningCSV"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = CICIDS2017_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
        sel_cols = [0, 1, 8, 10, 13, 24, 25, 31, 34, 41,  42, 44, 57, 65]
        X_train = train_x.iloc[:, sel_cols]
        X_valid = valid_x.iloc[:, sel_cols]
        X_test = test_x.iloc[:, sel_cols]
    elif data_name == "CICIDS2019":
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = CICIDS2019_processing(data_dir)
    elif data_name == "WSNDS":
        print("Preprocessing for WSNDS")
        data_dir = "./data/WSN-DS"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = WSNDS_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
        sel_cols = [0, 5, 6, 8, 9, 13, 14, 15, 17]
        X_train = train_x.iloc[:, sel_cols]
        X_valid = valid_x.iloc[:, sel_cols]
        X_test = test_x.iloc[:, sel_cols]
    elif data_name == "UNSW_NB15":
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = UNSW_NB15_processing(data_dir)
    elif data_name == "WSNBFSF":
        print("Preprocessing for WSNBFSF")
        data_dir = "./data/WSNBFSFDataset"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = WSNBFSF_processing(data_dir)
        train_y = le.fit_transform(train_y)
        valid_y = le.fit_transform(valid_y)
        test_y = le.fit_transform(test_y)
        sel_cols = [0, 3, 4, 10, 12]
        X_train = train_x.iloc[:, sel_cols]
        X_valid = valid_x.iloc[:, sel_cols]
        X_test = test_x.iloc[:, sel_cols]
    else:
        print("This type of data has not been processed yet")
    depth = int(depth)
    iterations = int(iterations)
    l2_leaf_reg = int(l2_leaf_reg)
    categorical_columns = list(train_x.select_dtypes(exclude=["number"]).columns)
    categorical_features_indices = train_x.columns.get_indexer(categorical_columns)
    # Initialize the CatBoost model (CatBoostRegressor for regression)
    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        random_strength = random_strength,
        bagging_temperature=2,
        verbose=False,
        cat_features= categorical_features_indices,
        bootstrap_type = "Bayesian",
        loss_function = 'MultiClass',
        eval_metric = 'Accuracy',
        random_seed = 42,
        od_type = 'Iter',
        od_wait = 20,
        task_type ="CPU"
    )
 
    # Perform cross-validation and return the mean R-squared score (for regression)
    cross_val_scores = cross_val_score(model, X_train, train_y, cv=3, scoring="accuracy")
 
    return cross_val_scores.mean()

# Define the hyperparameter search space with data types
param_space = {
    'depth': (3, 10),             # Integer values for depth
    'learning_rate': (0.01, 0.3),  # Float values for learning rate
    'iterations': (100, 1000),    # Integer values for iterations
    'random_strength': (0, 10),       # Float values for subsample
    'l2_leaf_reg': (1, 10)      # Integer values for l2_leaf_reg=
}
 
# Create the BayesianOptimization object and maximize it
bayesian_opt = BayesianOptimization(
    f=catboost_cv, pbounds=param_space, random_state=42)
bayesian_opt.maximize(init_points=5, n_iter=10)

# Print the best hyperparameters and their corresponding R2 score
best_hyperparameters = bayesian_opt.max
best_hyperparameters['params'] = {param: int(value) if param in [
    'depth', 'iterations', 'l2_leaf_reg'] else value for param, value in best_hyperparameters['params'].items()}
print("Best hyperparameters:", best_hyperparameters['params'])
print(f"Best R-squared Score: {best_hyperparameters['target']:.4f}")
