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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


columns = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
       'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
       'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
       'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
       'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
       'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
       'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
       'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
       'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
       'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
       'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
       'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
       'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
       'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
       'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

class CICIDS2019Preprocessor(object):
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
        datasets = [pd.read_csv(filename)[columns] for filename in filenames]

        # Remove white spaces and rename the columns
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # Concatenate the datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)

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
            'Benign': 'Benign',
            'DDoS attacks-LOIC-HTTP': 'PortScan&DDoS',
            'DDOS attack-HOIC': 'PortScan&DDoS',
            'DDOS attack-LOIC-UDP': 'PortScan&DDoS',
            'DoS attacks-GoldenEye': 'DoS/DDoS',                 
            'DoS attacks-Slowloris': 'DoS/DDoS',             
            'DoS attacks-Hulk': 'DoS/DDoS',        
            'DoS attacks-SlowHTTPTest': 'DoS/DDoS',        
            'DoS Slowhttptest': 'DoS/DDoS',    
            'Heartbleed': 'DoS/DDoS',    
            'FTP-BruteForce': 'Brute Force',           
            'SSH-Bruteforce': 'Brute Force',
            'Bot': 'Botnet attack',
            'Brute Force -Web': 'Web Attack',
            'SQL Injection': 'Web Attack',
            'Brute Force -XSS': 'Web Attack',
            'Infilteration': 'Infiltration'
        }

        # Create grouped label column
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
        self.data = self.data.drop(self.data[self.data['label_category']=="Benign"].sample(frac=0.82).index)
        
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
    
def CICIDS2019_processing(data_dir):
    cicids2019 = CICIDS2019Preprocessor(
        data_path=data_dir,
        training_size=0.6,
        validation_size=0.2,
        testing_size=0.2
    )

    # Read datasets
    cicids2019.read_data()

    # Remove NaN, -Inf, +Inf, Duplicates
    cicids2019.remove_duplicate_values()
    cicids2019.remove_missing_values
    cicids2019.remove_infinite_values()

    # Drop constant & correlated features
    cicids2019.remove_constant_features()
    # cicids2019.remove_correlated_features()
    # print(cicids2019.data.info())

    # Create new label category
    cicids2019.group_labels()
    

    # Split & Normalise data sets
    training_set, validation_set, testing_set            = cicids2019.train_valid_test_split()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cicids2019.scale(training_set, validation_set, testing_set)
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
        default="./data/CICIDS2019",
        required=False,
        help="Path to the data folder",
    )
    args = parser.parse_args()
    CICIDS2019_processing(args.data_dir)