import numpy as np
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
from catboost import metrics
from sklearn.metrics import accuracy_score, zero_one_loss, f1_score
from sklearn.model_selection import train_test_split

def quantize_to_binary_array(number, min_value, max_value, num_bits):
    """
    Quantize a number within a specified range into a binary NumPy array using a given number of bits.

    Args:
    number (float): The number to be quantized.
    min_value (float): The minimum value of the range.
    max_value (float): The maximum value of the range.
    num_bits (int): The number of bits in the binary representation.

    Returns:
    binary_array (numpy.ndarray): The binary NumPy array representing the quantized number.
    """
    
    # Ensure that the number is within the specified range
    number = max(min(number, max_value), min_value)
    
    # Calculate the range of values in the specified range
    value_range = max_value - min_value
    
    # Calculate the step size for each bit
    step_size = value_range / (2 ** num_bits)
    
    # Quantize the number to a binary NumPy array
    quantized_value = round((number - min_value) / step_size)
    binary_string = format(quantized_value, f'0{num_bits}b')
    binary_array = np.array([int(bit) for bit in binary_string])
    
    return binary_array

def dequantize_from_binary_array(binary_array, min_value, max_value):
    """
    Dequantize a binary NumPy array into a floating-point number within a specified range.

    Args:
    binary_array (numpy.ndarray): The binary NumPy array to be dequantized.
    min_value (float): The minimum value of the range.
    max_value (float): The maximum value of the range.

    Returns:
    dequantized_number (float): The dequantized number within the specified range.
    """
    
    # Calculate the step size for each bit
    num_bits = len(binary_array)
    value_range = max_value - min_value
    step_size = value_range / (2 ** num_bits)
    
    # Convert the binary NumPy array to an integer
    quantized_value = binary_array.dot(2 ** np.arange(num_bits)[::-1])
    
    # Dequantize the integer value to a floating-point number
    dequantized_number = min_value + quantized_value * step_size
    
    return dequantized_number

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
            xt,
            yt,
            test_size=0.2,
            shuffle=True
        )

    xtest  = xv
    ytest  = yv

    # Dequatization
    # print(f"length of parameters x:{len(x)} and x:{x}")
    iterations              = dequantize_from_binary_array(x[0:6], 50, 690)             # 4bits
    learning_rate           = dequantize_from_binary_array(x[6:20], 0.001, 0.5)         # 14bits
    depth                   = dequantize_from_binary_array(x[20:23], 1, 9)              # 3bits
    l2_leaf_reg             = dequantize_from_binary_array(x[23:26], 1, 9)              # 3bits
    random_strength         = dequantize_from_binary_array(x[26:34], 1e-8, 10)          # 8bits
    bagging_temperature     = dequantize_from_binary_array(x[34:42], 1e-8, 10)          # 8bits

    params = {"iterations": round(iterations), "learning_rate": learning_rate, "depth": round(depth),
              "l2_leaf_reg": l2_leaf_reg, "random_strength": random_strength, "bagging_temperature": bagging_temperature}
    # print(f"the parameters is: {params}")
    # Training new model 
    mdl = cb.CatBoostClassifier(
            learning_rate=learning_rate, 
            iterations=round(iterations), 
            depth=round(depth), 
            l2_leaf_reg=l2_leaf_reg, 
            bagging_temperature=bagging_temperature,
            random_seed=42,
            random_strength = random_strength,
            # boosting_type="Ordered", # Valid values: string, any of the following: ("Auto", "Ordered", "Plain").
            bootstrap_type="Bayesian",
            loss_function='MultiClass', 
            eval_metric="TotalF1",
            od_type='Iter',
            od_wait=20,
            task_type="GPU",
            )
    categorical_columns = list(xtrain.select_dtypes(exclude=["number"]).columns)
    # if categorical_columns:
    #     print(f"================there are {len(categorical_columns)} categorical columns=============") 
    # else:
    #     print("================No categorical columns=============")
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
    # Get error rate
    error = error_rate(xtrain, ytrain, x, opts)
    # Objective function
    # cost  = alpha * error + beta * (num_feat / max_feat)
    cost = error
        
    return cost