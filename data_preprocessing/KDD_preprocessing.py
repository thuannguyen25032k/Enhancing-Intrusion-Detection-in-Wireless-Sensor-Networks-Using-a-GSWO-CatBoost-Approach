# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:31:20 2023

@author: Nguyen Minh Thuan
"""
# module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from collections import defaultdict

# processing imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

print("Welcome!")
HEADER_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_type",
    "success_pred",
]


def load_data(data_path):
    """Loads the data from the specified path"""
    data = pd.read_csv(data_path)
    data.columns = HEADER_NAMES
    return data


def transform(data, attack_mapping):
    """Transforms the data into the format expected by the model"""
    data["attack_category"] = data["attack_type"].map(lambda x: attack_mapping[x])
    data_Y = data["attack_category"]
    data_X = data.drop(["attack_type", "attack_category"], axis=1)
    return data_X, data_Y


def scale(train_data_X, test_data_X, features, scaler):
    """Scales the data"""
    scaler.fit(train_data_X[features])
    train_data_X[features] = scaler.transform(train_data_X[features])
    test_data_X[features] = scaler.transform(test_data_X[features])
    return train_data_X, test_data_X


def visualize(transformed_data):
    """Visualizes the transformed data"""
    data_attack_types = transformed_data["attack_type"].value_counts()
    data_attack_cats = transformed_data["attack_category"].value_counts()
    data_attack_types.plot(kind="barh", figsize=(20, 10), fontsize=20)
    data_attack_cats.plot(kind="barh", figsize=(20, 10), fontsize=30)
    plt.show()

def shuffle_data(train_data, test_data):
    entire_data = pd.concat([train_data, test_data])
    train, test = train_test_split(entire_data, test_size=0.3, shuffle=True)
    return train, test

def split_data(X, y):
    """Split the data into train and validation"""
    X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )
    return (X_train, y_train), (X_valid, y_valid)

def KDD_processing(data_dir):
    column_names = np.array(HEADER_NAMES)

    # Differentiating between nominal, binary, and numeric features
    nominal_idx = [1, 2, 3]
    binary_idx = [6, 11, 13, 20, 21]
    numeric_idx = list(
        set(range(43)).difference(nominal_idx).difference(binary_idx).difference([41])
    )
    nominal_cols = column_names[nominal_idx].tolist()
    binary_cols = column_names[binary_idx].tolist()
    numeric_cols = column_names[numeric_idx].tolist()

    # training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories
    # file obtained from http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types
    category = defaultdict(list)
    category["benign"].append("normal")
    with open(os.path.join(data_dir, "training_attack_types.txt"), "r") as f:
        for line in f.readlines():
            attack, cat = line.strip().split(" ")
            category[cat].append(attack)

    attack_mapping = dict((v, k) for k in category for v in category[k])

    # Loading the data
    train_file_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_file_path = os.path.join(data_dir, "KDDTest+.txt")
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)
    train_data, test_data = shuffle_data(train_data, test_data)

    # Transforming the data
    train_data_X, train_data_Y = transform(train_data, attack_mapping)
    test_data_X, test_data_Y = transform(test_data, attack_mapping)

    # Visualizing the data
    # visualize(train_data)
    # visualize(test_data)

    # Analyzing the su_attempted column
    train_data.groupby(["su_attempted"]).size()

    # Analyzing the num_outbound_cmds column. we notice that the num_outbound_cmds column only takes on one value!
    train_data.groupby(["num_outbound_cmds"]).size()

    # Experimenting with StandardScaler on the single 'duration' feature
    durations = train_data_X["duration"].values.reshape(-1, 1)
    standard_scaler = StandardScaler().fit(durations)
    scaled_durations = standard_scaler.transform(durations)
    pd.Series(scaled_durations.flatten()).describe()

    # Experimenting with MinMaxScaler on the single 'duration' feature
    min_max_scaler = MinMaxScaler().fit(durations)
    min_max_scaled_durations = min_max_scaler.transform(durations)
    pd.Series(min_max_scaled_durations.flatten()).describe()

    # Experimenting with RobustScaler on the single 'duration' feature
    min_max_scaler = RobustScaler().fit(durations)
    robust_scaled_durations = min_max_scaler.transform(durations)
    pd.Series(robust_scaled_durations.flatten()).describe()

    # Let's proceed with StandardScaler- Apply to all the numeric columns
    scaler = StandardScaler()
    train_data_X, test_data_X = scale(train_data_X, test_data_X, numeric_cols, scaler)
    (X_train, y_train), (X_valid, y_valid) = split_data(train_data_X,train_data_Y)
    # print((X_train, y_train))
    X_train.index = pd.RangeIndex(0, len(X_train), step=1) 
    y_train.index = pd.RangeIndex(0, len(X_train), step=1)
    # print((X_train, y_train))
    return (X_train, y_train), (X_valid, y_valid), (test_data_X, test_data_Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing of the data")
    parser.add_argument(
        "--data_dir",
        "-dir",
        type=str,
        action="store",
        default="./data/NSL-KDD-Dataset",
        required=False,
        help="Path to the data folder",
    )
    args = parser.parse_args()
    KDD_processing(args.data_dir)
