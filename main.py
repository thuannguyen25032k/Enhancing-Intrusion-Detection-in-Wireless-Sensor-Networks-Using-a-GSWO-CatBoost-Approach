
import argparse
import numpy as np
import pandas as pd
from data_preprocessing import WSNDS_processing, KDD_processing, UNSW_NB15_processing, CICIDS2017_processing, WSNBFSF_processing, CICIDS2019_processing
from utils import TrainingClassifier
import smote_variants as sv
from sklearn.preprocessing import LabelEncoder
from feature_selection.wrapper_FS import FeatureSelector
from finetuning_parameters import finetuner
import logging
from multiprocessing import Pool

logger = logging.getLogger('my-logger')
logger.propagate = False

def oversample_data(data, categorical_features_list):
    """
    This function is used to oversample the data with:
        Input: data is training dataset including labels
                categorical_features_list is a list of categorical features
        Output: oversampled data
    """
    encode_dict = {}
    (X, y) = data
    for fe in categorical_features_list:
        encode_dict[fe] = LabelEncoder()
        X[fe] = encode_dict[fe].fit_transform(X[fe])
    data_columns = X.columns

    # oversampling the dataset
    oversampler= sv.MulticlassOversampling(oversampler='Borderline_SMOTE2',
                                      oversampler_params={'n_neighbors': 10, 
                                                          'k_neighbors': 10})
    # initial classification object
    X_samp, y_samp= oversampler.sample(X, y)
    oversampled_X_train = pd.DataFrame(X_samp, columns=data_columns)
    convert_dict = dict((k, int) for k in categorical_features_list)
    oversampled_X_train = oversampled_X_train.astype(convert_dict)
    for fe in categorical_features_list:
        oversampled_X_train[fe] = encode_dict[fe].inverse_transform(oversampled_X_train[fe]).astype(str)
    oversampled_y_train = pd.Series(y_samp,  name="attack_category")
    return (oversampled_X_train, oversampled_y_train)

def finetune(X_train, y_train, X_test, y_test):
    # prepare data feadfoward to Genetic Whale Optimization Algorithm
    fold = {'xt': X_train, 'yt':y_train, 'xv':X_test, 'yv':y_test}

    # parameters
    k    = 5     
    N    = 30    
    T    = 30   
    b    = 1
    CR   = 0.7
    MR   = 0.2    
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'b':b, 'CR':CR, 'MR':MR}
    fmdl = finetuner(X_train, y_train, opts)
    return fmdl["params"]

def main(data_name):
    """This function is responsible for main processing"""
    save_path = F"./figures/ROC CURVE FOR {data_name} DATASET.jpg"
    if data_name == "KDD":
        print("Preprocessing for KDD")
        data_dir = "./data/NSL-KDD-Dataset"
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = KDD_processing(data_dir)
        #seleced columns [ 0  3  4 10 12]
        # print(X_train.shape)
        sel_cols = [0, 2, 4, 5, 7, 13, 14, 15, 17, 18, 21, 22, 23, 25, 26, 27, 29, 32, 34, 36, 37, 38, 41]
        X_train = X_train.iloc[:, sel_cols]
        X_valid = X_valid.iloc[:, sel_cols]
        X_test = X_test.iloc[:, sel_cols]
        finetuned_params = {'iterations': 500, 'learning_rate': 0.05, 'depth': 2, 'l2_leaf_reg': 3, 'random_strength': 1, 'bagging_temperature': 1}
        # finetuned_params = {'iterations': 515, 'learning_rate': 0.3, 'depth': 6, 'l2_leaf_reg': 2, 'random_strength': 8.9, 'bagging_temperature': 3.3}
        
    elif data_name == "CICIDS2017":
        print("Preprocessing for CICIDS2017")
        data_dir = "./data/CICIDS17/MachineLearningCSV"
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = CICIDS2017_processing(data_dir)
        # print(X_train.shape)
        sel_cols = [0, 1, 8, 10, 13, 24, 25, 31, 34, 41,  42, 44, 57, 65]
        X_train = X_train.iloc[:, sel_cols]
        X_valid = X_valid.iloc[:, sel_cols]
        X_test = X_test.iloc[:, sel_cols]
        finetuned_params = {'iterations': 194, 'learning_rate': 0.2783, 'depth': 9, 'l2_leaf_reg': 2, 'random_strength': 7.707, 'bagging_temperature': 2}

    elif data_name == "CICIDS2019":
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = CICIDS2019_processing(data_dir)
    elif data_name == "WSNDS":
        print("Preprocessing for WSNDS")
        data_dir = "./data/WSN-DS"
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = WSNDS_processing(data_dir)
        # print(X_train.shape)
        sel_cols = [0, 5, 6, 8, 9, 13, 14, 15, 17]
        X_train = X_train.iloc[:, sel_cols]
        X_valid = X_valid.iloc[:, sel_cols]
        X_test = X_test.iloc[:, sel_cols]
        finetuned_params = {'iterations': 370, 'learning_rate': 0.3033, 'depth': 6, 'l2_leaf_reg': 2.0, 'random_strength': 8.9063, 'bagging_temperature': 0.7031}

    elif data_name == "UNSW_NB15":
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = UNSW_NB15_processing(data_dir)

    elif data_name == "WSNBFSF":
        print("Preprocessing for WSNBFSF")
        data_dir = "./data/WSNBFSFDataset"
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = WSNBFSF_processing(data_dir)
        # print(X_train.shape)
        sel_cols = [0, 3, 4, 10, 12]
        X_train = X_train.iloc[:, sel_cols]
        X_valid = X_valid.iloc[:, sel_cols]
        X_test = X_test.iloc[:, sel_cols]
        finetuned_params = {'iterations': 270, 'learning_rate':  0.28817, 'depth': 5, 'l2_leaf_reg': 2.0, 'random_strength': 0.15625, 'bagging_temperature': 1.09375}

    else:
        print("This type of data has not been processed yet")

    # implementing the classification algorithm
    categorical_columns = list(X_train.select_dtypes(exclude=["number"]).columns)
    categorical_features_indices = X_train.columns.get_indexer(categorical_columns)
    X_train.index = pd.RangeIndex(0, len(X_train), step=1) 
    y_train.index = pd.RangeIndex(0, len(X_train), step=1)

    # implementing feature selection
    # selector = FeatureSelector(X_train.copy(), y_train.copy(), X_valid.copy(), y_valid.copy())
    # X_train, y_train, X_valid, y_valid = selector.BA()
    # X_test = X_test.iloc[:, selector.sel_col]
    # print(f"Useful feature: {selector.sel_col}")

    # result test: 
    # sel_col = [ 1,  2,  4,  8, 10, 11, 16, 18, 19, 21, 22, 24, 25, 26, 27, 29, 30, 31, 32, 35, 37, 39, 40, 41]

    # # # print(oversampled_y_train.value_counts())
    # print(X_train.shape)
    # categorical_columns = list(X_train.select_dtypes(exclude=["number"]).columns)
    # categorical_features_indices = X_train.columns.get_indexer(categorical_columns)
    # finetuned_params = finetune(X_train.copy(), y_train.copy(), X_valid.copy(), y_valid.copy())
    # finetuned_params = {'iterations': 500, 'learning_rate': 0.05, 'depth': 2, 'l2_leaf_reg': 3.0, 'random_strength': 1, 'bagging_temperature': 1}
    # finetuned_params = {'iterations': 220, 'learning_rate':  0.07, 'depth': 10, 'l2_leaf_reg': 3, 'random_strength':  5, 'bagging_temperature':  5}
    # classifier = TrainingClassifier({"cat_features": categorical_features_indices})
    # finetuned_params = classifier.grid_search((X_train, y_train), (X_valid, y_valid))
    print(finetuned_params)
    classifier = TrainingClassifier(finetuned_params)
    classifier.train((X_train, y_train), (X_valid, y_valid), categorical_features_indices)
    classifier.evaluate((X_test, y_test), save_path)
    # print(f"Useful feature: {selector.sel_col}")
    # classifier.visulize_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main program")
    parser.add_argument(
        "--data_name",
        "-n",
        type=str,
        action="store",
        default="KDD",
        required=True,
        help="Data names are in upper case format",
    )
    parser.add_argument(
        "--NSLKDD_data_dir",
        "-k",
        type=str,
        action="store",
        default="./data/NSL-KDD-Dataset",
        required=False,
        help="Path to the data folder",
    )
    parser.add_argument(
        "--CICIDS2017_data_dir",
        "-ci17",
        type=str,
        action="store",
        default="./data/CICIDS17/MachineLearningCSV",
        required=False,
        help="Path to the data folder",
    )
    parser.add_argument(
        "--CICIDS2019_data_dir",
        "-ci18",
        type=str,
        action="store",
        default="./data/CICIDS2019",
        required=False,
        help="Path to the data folder",
    )
    parser.add_argument(
        "--WSNDS_data_dir",
        "-ws",
        type=str,
        action="store",
        default="./data/WSN-DS",
        required=False,
        help="Path to the data folder",
    )
    parser.add_argument(
        "--UNSWNB15_data_dir",
        "-un",
        type=str,
        action="store",
        default="./data/UNSW-NB15-Dataset",
        required=False,
        help="Path to the data folder",
    )
    parser.add_argument(
        "--WSNBFSF_data_dir",
        "-bf",
        type=str,
        action="store",
        default="./data/WSNBFSFDataset",
        required=False,
        help="Path to the data folder",
    )
    args = parser.parse_args()
    if args.data_name == "KDD":
        with Pool(2) as p:
            p.map(main, [args.data_name])
    elif args.data_name == "CICIDS2017":
        with Pool(2) as p:
            p.map(main, [args.data_name])
    elif args.data_name == "CICIDS2019":
        main(args.data_name, args.CICIDS2019_data_dir)
    elif args.data_name == "WSNDS":
        with Pool(2) as p:
            p.map(main, [args.data_name])
    elif args.data_name == "UNSW_NB15":
        main(args.data_name, args.UNSWNB15_data_dir)
    elif args.data_name == "WSNBFSF":
        with Pool(2) as p:
            p.map(main, [args.data_name])
            # main(args.data_name, args.WSNBFSF_data_dir)
    else:
        print("This type of data has not been processed yet")

