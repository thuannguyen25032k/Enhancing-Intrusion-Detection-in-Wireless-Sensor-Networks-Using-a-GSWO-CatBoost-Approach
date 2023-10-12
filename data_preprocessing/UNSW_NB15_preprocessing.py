# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:31:20 2023

@author: Nguyen Minh Thuan
"""
# module imports
import argparse
import os
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

# processing imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Welcome!")


def load_data(data_dir, name):
    """Load data."""
    data = pd.read_csv(os.path.join(data_dir, name + ".csv"))
    return data


def convert_to_nummerics(data, features):
    """Convert features to numerics."""
    # selecting column names of all data types
    nominal_names = features["Name"][features["Type "] == "nominal"]
    integer_names = features["Name"][features["Type "] == "integer"]
    binary_names = features["Name"][features["Type "] == "binary"]
    float_names = features["Name"][features["Type "] == "float"]

    # selecting common column names from dataset and feature dataset
    cols = data.columns
    nominal_names = cols.intersection(nominal_names)
    integer_names = cols.intersection(integer_names)
    binary_names = cols.intersection(binary_names)
    float_names = cols.intersection(float_names)

    # Converting integer columns to numeric
    for i in integer_names:
        pd.to_numeric(data[i])
    # Converting binary columns to numeric
    for b in binary_names:
        pd.to_numeric(data[b])
    # Converting float columns to numeric
    for f in float_names:
        pd.to_numeric(data[f])
    return data


def scale(train_data_X, test_data_X, features, scaler):
    """Scales the data"""
    scaler.fit(train_data_X[features])
    train_data_X[features] = scaler.transform(train_data_X[features])
    test_data_X[features] = scaler.transform(test_data_X[features])
    return train_data_X, test_data_X


def transform(data):
    """Transforms the data into the format expected by the model"""
    data_Y = data["attack_cat"]
    data_X = data.drop(["label", "attack_cat"], axis=1)
    # print(data_Y)
    return data_X, data_Y


def visualize(data):
    """Visualizes the data"""
    plt.figure(figsize=(8, 8))
    plt.pie(
        data.attack_cat.value_counts(),
        labels=data.attack_cat.unique(),
        autopct="%0.2f%%",
    )
    plt.title("Pie chart distribution of multi-class labels")
    plt.legend(loc="best")
    plt.savefig("plots/Pie_chart_multi.png")
    plt.show()

def split_data(X, y):
    """Split the data into train and validation"""
    X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    return (X_train, y_train), (X_valid, y_valid)

def UNSW_NB15_processing(data_dir):
    """Preprocessing of the UNSW_NB15 data"""
    # loading data
    raw_train_data = load_data(data_dir, "UNSW_NB15_training-set")
    raw_train_data['service']= np.where(raw_train_data['service'] == '-', 'other', raw_train_data['service'])
    raw_test_data = load_data(data_dir, "UNSW_NB15_testing-set")
    raw_test_data['service']= np.where(raw_test_data['service'] == '-', 'other', raw_test_data['service'])
    # features = load_data(data_dir, "UNSW_NB15_features")
    # print(features)
    # train_data = convert_to_nummerics(raw_train_data, features)
    # test_data = convert_to_nummerics(raw_test_data, features)

    # transforming data into the format expected by the model
    train_data_X, train_data_Y = transform(raw_train_data.drop("id", axis=1))
    test_data_X, test_data_Y = transform(raw_test_data.drop("id", axis=1))

    # visualize data
    # visualize(raw_train_data)
    # visualize(raw_train_data)

    # selecting numeric attributes columns from data
    numeric_cols = list(raw_train_data.select_dtypes(include="number").columns)
    numeric_cols.remove("id")
    numeric_cols.remove("label")
    scaler = StandardScaler()
    train_data_X, test_data_X = scale(train_data_X, test_data_X, 
                                      numeric_cols, scaler)
    (X_train, y_train), (X_valid, y_valid) = split_data(train_data_X,train_data_Y)
    return (X_train, y_train), (X_valid, y_valid), (test_data_X, test_data_Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing of the data")
    parser.add_argument(
        "--data_dir",
        "-dir",
        type=str,
        action="store",
        default="./data/UNSW-NB15-Dataset",
        required=False,
        help="Path to the data folder",
    )
    args = parser.parse_args()
    UNSW_NB15_processing(args.data_dir)
