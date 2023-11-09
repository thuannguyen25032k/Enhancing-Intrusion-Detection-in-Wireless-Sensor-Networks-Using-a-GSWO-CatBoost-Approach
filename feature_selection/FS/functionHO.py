import numpy as np
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, zero_one_loss, f1_score
from main import oversample_data
# import xgboost as xgb

# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    k     = opts['k']
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']

    # Define selected features using dataframe for xtrain and ytrain, xvalid and yvalid
    xtrain, xvalid, ytrain, yvalid = train_test_split(
            xt.iloc[:, x==1],
            yt,
            test_size=0.2,
            shuffle=True
        )

    xtest  = xv.iloc[:, x==1]
    ytest  = yv

    # Training
    # mdl     = RandomForestClassifier(criterion='gini', max_depth=30, n_estimators=72, random_state=0, bootstrap=False, min_samples_split = 2)
    
    # Training new model 
    params = {'iterations': 120, 'learning_rate': 0.12575, 'depth': 2, 'l2_leaf_reg': 2.0, 'random_strength': 1e-08, 'bagging_temperature': 1e-08,
              'bootstrap_type': "Bayesian", "loss_function": 'MultiClass', "eval_metric":"TotalF1", "od_type": 'Iter', "od_wait":20,
                "task_type":"GPU"}
    mdl = cb.CatBoostClassifier(**params)
    categorical_columns = list(xtrain.select_dtypes(exclude=["number"]).columns)
    categorical_features_indices = xtrain.columns.get_indexer(categorical_columns)
    
    # fit model
    mdl.fit(
            xtrain, ytrain, 
            cat_features= categorical_features_indices,
            eval_set=(xvalid, yvalid),
            logging_level='Silent',
            )
    
    # prediction 
    ypred   = mdl.predict(xtest)

    error = zero_one_loss(ytest, ypred)
    # error = 1- f1_score(yvalid, ypred, average='micro') 
    
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

