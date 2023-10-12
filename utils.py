import numpy as np
from catboost import CatBoostClassifier, Pool, metrics, cv
import category_encoders as ce 
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, zero_one_loss

class TrainingClassifier(object):
    def __init__(self):
        self.model = CatBoostClassifier(
            learning_rate=0.3, 
            iterations=500, 
            max_depth=4, 
            l2_leaf_reg=1, 
            bagging_temperature=0.5,
            use_best_model=True,
            # boosting_type="Ordered", # Valid values: string, any of the following: ("Auto", "Ordered", "Plain").
            bootstrap_type="Bayesian",
            loss_function='MultiClass', 
            eval_metric=metrics.Accuracy(),
            od_type='Iter',
            od_wait=20,
            )
        self.cbe_encoder = ce.cat_boost.CatBoostEncoder() 

        self.classes = None
        self.testing_data = None
        self.accuracy = None
        self.error = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def train(self, traning_data, validation_data, categorical_features_indices):
        ''' 
        Train the CatBoost classifier with:
            Training_data includes (X_train, Y_train)
            Validation_data includes (X_val, Y_val)
            categorical_features_indices is the list of number indicating the index of categorical featurers in dataset
        '''
        (X_train, y_train) = traning_data
        self.model.fit(
            X_train, y_train, 
            cat_features= categorical_features_indices,
            eval_set= validation_data,
            logging_level='Verbose', #Silent 
            )
        
    def predict(self, X_test):
        """ This function implements prediction based on X with:
            input: X_test
            output: predictions
        """
        predictions = self.model.predict(X_test)
        return predictions
    
    def save(self, filepath):
        """This function implements storing a model"""
        self.model.save_model(filepath)

    def load(self, filepath):
        self.model.load_model(filepath)

    def evaluate(self, testing_data):
        """This function tests model"""
        self.testing_data = testing_data
        (X_test, y_test) = testing_data
        self.classes = np.unique(y_test)
        y_pred = self.predict(X_test)
        self.precision = precision_score(y_test, y_pred, average=None)  # 'macro', 'micro', 'weighted'
        self.recall = recall_score(y_test, y_pred, average=None)
        self.f1 = f1_score(y_test, y_pred, average=None)
        self.error = zero_one_loss(y_test, y_pred)
        print("Error: ", self.error)
        self.accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", self.accuracy)
        confusion_mat = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", confusion_mat)
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:\n", class_report)

    def visulize_results(self):
        # Create bar plots for precision, recall, and F1-score
        fig, ax = plt.subplots(3, 1, figsize=(8, 12))

        # Precision
        ax[0].bar(self.classes, self.precision, tick_label=self.classes)
        ax[0].set_title('Precision per Class')
        ax[0].set_xlabel('Class')
        ax[0].set_ylabel('Precision')

        # Recall
        ax[1].bar(self.classes, self.recall, tick_label=self.classes)
        ax[1].set_title('Recall per Class')
        ax[1].set_xlabel('Class')
        ax[1].set_ylabel('Recall')

        # F1-score
        ax[2].bar(self.classes, self.f1, tick_label=self.classes)
        ax[2].set_title('F1-score per Class')
        ax[2].set_xlabel('Class')
        ax[2].set_ylabel('F1-score')

        plt.tight_layout()

        # Create line plots for precision and recall
        plt.figure(figsize=(8, 6))

        # Precision
        plt.plot(self.classes, self.precision, marker='o', label='Precision', linestyle='-', color='b')

        # Recall
        plt.plot(self.classes, self.recall, marker='o', label='Recall', linestyle='-', color='r')

        plt.title('Precision and Recall per Class')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(self.classes)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.show()

    def encode_feature(self, dataset):
        """This function encodes the categorical features"""
        (data,labels) = dataset
        self.cbe_encoder.fit(data,labels)
        data_cbe = self.cbe_encoder.transform(data)
        return (data_cbe, labels)


        