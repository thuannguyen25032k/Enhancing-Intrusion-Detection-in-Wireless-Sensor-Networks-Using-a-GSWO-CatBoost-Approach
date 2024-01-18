# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:31:20 2023

@author: Nguyen Minh Thuan
"""
import pandas as pd
import numpy as np
import glob
import os
import argparse

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


# DATA_DIR  = os.path.join(os.path.abspath("."), "data")

class CICIDS2017Preprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        
        self.data = None
        self.features = None
        self.label = None

    def read_data(self):
        """"""
        filenames = glob.glob(os.path.join(self.data_path,'*.csv'))
        datasets = [pd.read_csv(filename) for filename in filenames]

        # Remove white spaces and rename the columns
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # Concatenate the datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        self.data.drop(labels=['fwd_header_length.1'], axis= 1, inplace=True)

    def _clean_column_name(self, column):
        """"""
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column

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

    def remove_correlated_features(self, threshold=0.99):
        """"""
        # Correlation matrix
        data_corr = self.data.corr()

        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find Features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        """"""
        # Proposed Groupings
        attack_group = {
            'BENIGN': 'Benign',
            'PortScan': 'PortScan',
            'DDoS': 'DoS/DDoS',
            'DoS Hulk': 'DoS/DDoS',
            'DoS GoldenEye': 'DoS/DDoS',
            'DoS slowloris': 'DoS/DDoS', 
            'DoS Slowhttptest': 'DoS/DDoS',
            'Heartbleed': 'DoS/DDoS',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Botnet ARES',
            'Web Attack � Brute Force': 'Web Attack',
            'Web Attack � Sql Injection': 'Web Attack',
            'Web Attack � XSS': 'Web Attack',
            'Infiltration': 'Infiltration'
        }

        # attack_group = {
        #     'BENIGN': 'Benign',
        #     'PortScan': 'PortScan',
        #     'DDoS': 'DDoS',
        #     'DoS Hulk': 'DoS Hulk',
        #     'DoS GoldenEye': 'DoS GoldenEye',
        #     'DoS slowloris': 'DoS slowloris', 
        #     'DoS Slowhttptest': 'DoS Slowhttptest',
        #     'Heartbleed': 'Heartbleed',
        #     'FTP-Patator': 'FTP-Patator',
        #     'SSH-Patator': 'SSH-Patator',
        #     'Bot': 'Botnet ARES',
        #     'Web Attack � Brute Force': 'Web Attack Brute Force',
        #     'Web Attack � Sql Injection': 'Web Attack Sql Injection',
        #     'Web Attack � XSS': 'Web Attack XSS',
        #     'Infiltration': 'Infiltration'
        # }

        # Create grouped label column
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
        self.data = self.data.drop(self.data[self.data['label_category']=="Benign"].sample(frac=0.84).index)
        
    def train_valid_test_split(self):
        """"""
        self.labels = self.data['label_category']
        self.features = self.data.drop(labels=['label', 'label_category'], axis=1)

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
        print(f"y_total: {self.labels.value_counts()}, y_test: {y_test.value_counts()}")
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
    
def CICIDS2017_processing(data_dir):
    cicids2017 = CICIDS2017Preprocessor(
        data_path=os.path.join(data_dir, "MachineLearningCVE"),
        training_size=0.5,
        validation_size=0.2,
        testing_size=0.3
    )

    # Read datasets
    cicids2017.read_data()

    # Remove NaN, -Inf, +Inf, Duplicates
    cicids2017.remove_duplicate_values()
    cicids2017.remove_missing_values
    cicids2017.remove_infinite_values()

    # Drop constant & correlated features
    cicids2017.remove_constant_features()
    # cicids2017.remove_correlated_features()
    # print(cicids2017.data.info())

    # Create new label category
    cicids2017.group_labels()
    

    # Split & Normalise data sets
    training_set, validation_set, testing_set            = cicids2017.train_valid_test_split()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cicids2017.scale(training_set, validation_set, testing_set)
    X_train.index = pd.RangeIndex(0, len(X_train), step=1) 
    y_train.index = pd.RangeIndex(0, len(X_train), step=1)
    # print(X_test, y_test)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocessing of the data")
    parser.add_argument(
        "--data_dir",
        "-dir",
        type=str,
        action="store",
        default="./data/CICIDS17/MachineLearningCSV",
        required=False,
        help="Path to the data folder",
    )
    args = parser.parse_args()
    CICIDS2017_processing(args.data_dir)