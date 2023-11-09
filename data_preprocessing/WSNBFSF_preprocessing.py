# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03rd 16:31:20 2023

@author: Nguyen Minh Thuan
"""
import pandas as pd
import numpy as np
import glob
import os
import argparse
import matplotlib.pyplot as plt

# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


# DATA_DIR  = os.path.join(os.path.abspath("."), "data")

class WSNBFSFPreprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        
        self.data = None
        self.features = None
        self.label = None

    def read_data(self):
        """Read the data from the file."""
        self.data = pd.read_csv(self.data_path)

    def visualize(self):
        """Visualize the data."""
        self.data["Class"].value_counts().plot.bar()  
        print(self.data)
        print(self.data.columns) 
        # plt.show()     

    def remove_duplicate_values(self):
        """"""
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        """"""
        # Remove missing values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        """"""
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        self.data.dropna(axis=0, how='any', inplace=True)

    def remove_constant_features(self, threshold=0.01):
        """"""
        # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
        data_std = self.data.std(numeric_only=True)

        # Find Features that meet the threshold
        constant_features = [column for column, std in data_std.items() if std < threshold]

        # Drop the constant features
        self.data.drop(labels=constant_features, axis=1, inplace=True)
        
    def train_valid_test_split(self):
        """"""
        self.labels = self.data['Class']
        self.features = self.data.drop(labels=['Class'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.labels,
            test_size=(self.validation_size + self.testing_size),
            random_state=42,
            stratify=self.labels
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=self.testing_size / (self.validation_size + self.testing_size),
            random_state=42
        )
    
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def scale(self, training_set, validation_set, testing_set):
        """"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set
        
        categorical_features = self.features.select_dtypes(exclude=["number"]).columns
        numeric_features = self.features.select_dtypes(exclude=[object]).columns

        preprocessor = ColumnTransformer(transformers=[
            # ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),
            ('numericals', StandardScaler(), numeric_features)
        ])

        # Preprocess the features
        columns = numeric_features.tolist()

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=columns)
        X_val = pd.DataFrame(preprocessor.transform(X_val), columns=columns)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=columns)

        # Preprocess the labels
        # le = LabelEncoder()

        # y_train = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
        # y_val = pd.DataFrame(le.transform(y_val), columns=["label"])
        # y_test = pd.DataFrame(le.transform(y_test), columns=["label"])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
def WSNBFSF_processing(data_dir):
    WSNBFSF = WSNBFSFPreprocessor(
        data_path=os.path.join(data_dir, "WSNBFSFdataset.csv"),
        training_size=0.6,
        validation_size=0.2,
        testing_size=0.2
    )

    # Read datasets
    WSNBFSF.read_data()

    # Remove NaN, -Inf, +Inf, Duplicates
    WSNBFSF.remove_duplicate_values()
    WSNBFSF.remove_missing_values()
    WSNBFSF.remove_infinite_values()

    # Drop constant & correlated features
    WSNBFSF.remove_constant_features()
    # # print(cicids2017.data.info())    

    # Split & Normalise data sets
    training_set, validation_set, testing_set            = WSNBFSF.train_valid_test_split()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = WSNBFSF.scale(training_set, validation_set, testing_set)
    # print(X_train, y_train)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocessing of the data")
    parser.add_argument(
        "--data_dir",
        "-dir",
        type=str,
        action="store",
        default="./data/WSNBFSFDataset",
        required=False,
        help="Path to the data folder",
    )
    args = parser.parse_args()
    WSNBFSF_processing(args.data_dir)