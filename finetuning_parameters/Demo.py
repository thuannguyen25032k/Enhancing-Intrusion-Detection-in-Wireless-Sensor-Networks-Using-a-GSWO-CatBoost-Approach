import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from finetuner import finetuner   # change this to switch algorithm 
import matplotlib.pyplot as plt


# load data
data  = pd.read_csv('./feature_selection/ionosphere.csv')

# print(np.random.randint(3, size=351).shape)
feat  = data.iloc[:, 0:-1]
label = data.iloc[:, -1]
# label = np.random.randint(5, size=351)
print(type(feat))

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
print(type(xtrain))
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 50    # number of chromosomes
T    = 100   # maximum number of generations
CR   = 0.8
MR   = 0.01
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR}

# perform feature selection
fmdl = finetuner(feat, label, opts)
# sf   = fmdl['sf']

# # model with selected features
# num_train = np.size(xtrain, 0)
# num_valid = np.size(xtest, 0)
# x_train   = xtrain[:, sf]
# y_train   = ytrain.reshape(num_train)  # Solve bug
# x_valid   = xtest[:, sf]
# y_valid   = ytest.reshape(num_valid)  # Solve bug

# mdl       = KNeighborsClassifier(n_neighbors = k) 
# mdl.fit(x_train, y_train)

# # accuracy
# y_pred    = mdl.predict(x_valid)
# Acc       = np.sum(y_valid == y_pred)  / num_valid
# print("Accuracy:", 100 * Acc)

# # number of selected features
# num_feat = fmdl['nf']
# print("Feature Size:", num_feat)

# # plot convergence
# curve   = fmdl['c']
# curve   = curve.reshape(np.size(curve,1))
# x       = np.arange(0, opts['T'], 1.0) + 1.0

# fig, ax = plt.subplots()
# ax.plot(x, curve, 'o-')
# ax.set_xlabel('Number of Iterations')
# ax.set_ylabel('Fitness')
# ax.set_title('GA')
# ax.grid()
# plt.show()