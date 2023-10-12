
import argparse
import numpy as np
import pandas as pd
from data_preprocessing import WSNDS_processing, KDD_processing, UNSW_NB15_processing, CICIDS2017_processing
from utils import TrainingClassifier
import smote_variants as sv
from sklearn.preprocessing import LabelEncoder

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
    X_samp, y_samp= oversampler.sample(X, y)
    oversampled_X_train = pd.DataFrame(X_samp, columns=data_columns)
    convert_dict = dict((k, int) for k in categorical_features_list)
    oversampled_X_train = oversampled_X_train.astype(convert_dict)
    for fe in categorical_features_list:
        oversampled_X_train[fe] = encode_dict[fe].inverse_transform(oversampled_X_train[fe]).astype(str)
    oversampled_y_train = pd.Series(y_samp,  name="attack_category")
    return (oversampled_X_train, oversampled_y_train)

def main(data_name, data_dir):
    """This function is responsible for main processing"""
    if data_name == "KDD":
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = KDD_processing(data_dir)
    elif data_name == "CICIDS2017":
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = CICIDS2017_processing(data_dir)
    elif data_name == "WSNDS":
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = WSNDS_processing(data_dir)
    elif data_name == "UNSW_NB15":
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = UNSW_NB15_processing(data_dir)
    else:
        print("This type of data has not been processed yet")

    # initial classification object
    y_train.index = pd.RangeIndex(0, len(y_train), step=1)
    classifier = TrainingClassifier() 
    print(y_test.value_counts())
    print(y_train.value_counts())

    # implementing the classification algorithm
    categorical_columns = list(X_train.select_dtypes(exclude=["number"]).columns)
    if categorical_columns:
        print(categorical_columns)
    else:
        print(y_train.value_counts())
        print("================No categorical columns=============")
    categorical_features_indices = X_train.columns.get_indexer(categorical_columns)
    # (oversampled_X_train, oversampled_y_train) = oversample_data((X_train, y_train), categorical_columns)
    # print(oversampled_y_train.value_counts())
    # classifier.train((oversampled_X_train, oversampled_y_train), (X_valid, y_valid), categorical_features_indices)
    classifier.train((X_train, y_train), (X_valid, y_valid), categorical_features_indices)
    classifier.evaluate((X_valid, y_valid))
    classifier.visulize_results()

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
        "-ci",
        type=str,
        action="store",
        default="./data/CICIDS17/MachineLearningCSV",
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
    args = parser.parse_args()
#     KDD_processing(args.NSLKDD_data_dir)
    main(args.data_name, args.UNSWNB15_data_dir)

