import numpy as np
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
from sklearn.metrics import accuracy_score
import xgboost as xgb
from utils import TrainingClassifier

# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    k     = opts['k']
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']
    
    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features using numpy arrays for xtrain and ytrain, xvalid and yvalid
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug 

    # Dèine selected features using dataframe for xtrain and ytrain, xvalid and yvalid
    # xtrain  = xt.iloc[:, x==1]
    # ytrain  = yt
    # xvalid  = xv.iloc[:, x==1]
    # yvalid  = yv

    # Training
    # mdl     = RandomForestClassifier(criterion='gini', max_depth=30, n_estimators=72, random_state=0, bootstrap=False, min_samples_split = 2)
    
    # Training new model 
    mdl = TrainingClassifier() 

    
    # print(xtrain)
    mdl.fit(xtrain, ytrain, verbose=False)
    # Prediction
    ypred   = mdl.predict(xvalid)
    # print("==========ypred=========:", ypred)
    # print("==========yvalid=========:", yv)
    # acc     = np.sum(yvalid == ypred) / num_valid
    acc = accuracy_score(yvalid, ypred)
    # print(acc)
    error   = 1 - acc
    
    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost  = alpha * error + beta * (num_feat / max_feat)
        
    return cost

