import numpy as np
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import category_encoders as ce 
from matplotlib import pyplot as plt
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, zero_one_loss

class TrainingClassifier(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.parameters.update({# 'use_best_model': True,
                                'bootstrap_type': "Bayesian",
                                'loss_function': 'MultiClass',
                                'eval_metric': 'Accuracy',
                                "random_seed": 42,
                                "od_type": 'Iter',
                                'od_wait': 20,
                                "task_type":"CPU"})
        self.model = CatBoostClassifier(
            **self.parameters
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
        # print(traning_data)
        self.model.fit(
            X_train, y_train, 
            cat_features= categorical_features_indices,
            eval_set= validation_data,
            logging_level='Silent', #Silent 
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

    def evaluate(self, testing_data, save_path):
        """This function tests model"""
        self.testing_data = testing_data
        (X_test, y_test) = testing_data
        self.classes = np.unique(y_test)
        start = time.time()
        y_pred = self.predict(X_test)
        print(f"Time for prediction: {time.time()-start}")
        self.precision = precision_score(y_test, y_pred, average='macro')  # 'macro', 'micro', 'weighted'
        self.recall = recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Precision score: {self.precision}, recall score: {self.recall}, F1 score:{self.f1}")
        self.error = zero_one_loss(y_test, y_pred)
        print("Error: ", self.error)
        self.accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", self.accuracy)
        confusion_mat = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", confusion_mat)
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:\n", class_report)
        print(f"y_test: {y_test.values}, y_pred: {y_pred[:, 0]}, self.classes: {self.classes}")
        self.visualizing_ROC_Curve(y_test.values, y_pred[:, 0], self.classes, save_path=save_path)
        

    def __print_cv_summary(self, cv_data):
        # cv_data.head(10)

        best_value = cv_data['test-Accuracy-mean'].max()
        best_iter = cv_data['test-Accuracy-mean'].values.argmax()

        print('Best validation Accuracy score : {:.4f}±{:.4f} on step {}'.format(
            best_value,
            cv_data['test-Accuracy-std'][best_iter],
            best_iter)
        )

    def cross_validate(self, whole_data, categorical_features_indices):
        (X,y) = whole_data
        train_pool = Pool(data=X, label=y, cat_features=categorical_features_indices, has_header=True)
        cv_data = cv(
                params = self.parameters,
                pool = train_pool,
                fold_count=10,
                shuffle=True,
                partition_random_seed=0,
                plot=True,
                stratified=True,
                verbose=False
                )
        print(cv_data.T)
        self.__print_cv_summary(cv_data)

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

    def grid_search(self, traning_data, validation_data):
        (X_train, y_train) = traning_data
        grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}
        grid_search_result = self.model.grid_search(grid, 
                                                    X=X_train, 
                                                    y=y_train, verbose=False
                                                    )
        print(grid_search_result['params'])
        return grid_search_result['params']

    def random_search(self, traning_data, validation_data):
        (X_train, y_train) = traning_data
        grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}
        randomize_search_result = self.model.randomized_search(grid, 
                                                    X=X_train, 
                                                    y=y_train, verbose=False)
        print(randomize_search_result['params'])
        print(randomize_search_result['best_params'])
        return randomize_search_result
    
    def visualizing_ROC_Curve(self, y_true, y_predict, classes, font_size=12, save_path=None):
        """
        Plot ROC curves for each class and micro-average ROC curve for multiclass classification.

        Parameters:
        - y_true: true labels (as strings)
        - y_predict: predicted labels or probabilities for each class (as strings)
        - classes: list of class labels

        Returns:
        - None (displays the plot)
        """

        # Convert string labels to numerical indices
        class_to_index = {label: i for i, label in enumerate(classes)}
        y_true_indices = np.array([class_to_index[label] for label in y_true])
        # print([class_to_index[label] for label in y_predict])
        y_predict_indices = np.array([class_to_index[label] for label in y_predict])

        # Binarize the labels if y_predict contains probabilities
        if y_predict.ndim > 1 and y_predict.shape[1] > 1:
            y_predict_bin = y_predict
        else:
            y_predict_bin = label_binarize(y_predict_indices, classes=np.arange(len(classes)))

        # Binarize the true labels
        y_true_bin = label_binarize(y_true_indices, classes=np.arange(len(classes)))
        n_classes = len(classes)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_predict_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_predict_bin.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot ROC curves
        plt.figure(figsize=(10, 6))

        # Plot individual class curves
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve ({classes[i]}) (AUC = {roc_auc[i]:.2f})')

        # Plot micro-average curve
        plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                linestyle=':', linewidth=4)

        # Plot macro-average curve
        plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
                linestyle='--', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # Random guess line

        plt.xlabel('False Positive Rate', fontsize=font_size)
        plt.ylabel('True Positive Rate', fontsize=font_size)
        plt.title(f'Receiver Operating Characteristic (ROC) Curves (Multiclass)', fontsize=font_size)
        plt.legend(loc="lower right", fontsize=font_size)

        if save_path:
            plt.savefig(save_path, format='jpg', bbox_inches='tight')
        else:
            plt.show()