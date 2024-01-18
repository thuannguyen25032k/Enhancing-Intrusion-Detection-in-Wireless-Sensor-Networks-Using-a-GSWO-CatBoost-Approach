import numpy as np
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
from numpy.random import rand
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
    
    # Dequantize the integer value tt number
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
    depth                   = dequantize_from_binary_array(x[20:23], 2, 9)              # 3bits
    l2_leaf_reg             = dequantize_from_binary_array(x[23:26], 2, 9)              # 3bits
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
            eval_metric="Accuracy",
            od_type='Iter',
            od_wait=20,
            task_type="CPU",
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

def hamming_distance(b1,b2):
    ans = 0
    for i in range(len(b1)):
        ans += not(b1[i]==b2[i])
    return ans

def similarity(beta, chromosome1, chromosome2, acc1, acc2):
    H_d = hamming_distance(chromosome1,chromosome2)
    D_a = abs(acc1-acc2)
    if (H_d !=0):
        S = 1/(H_d + D_a)
    else :
        S = 99999
    return S

def get_closest_ind(pop, acc, beta=0.3):
    ind1 = pop[0]
    acc1 = acc[0]
    similarity_list = []
    for i in range(1,len(pop)):
        ind2 = pop[i]
        acc2 = acc[i]
        similarity_list.append(similarity(beta, ind1, ind2, acc1, acc2))
    max_sim_index = similarity_list.index(max(similarity_list))+1
    # 1 is added to the index since the 1st item in similarity_index_list corresponds to individual number 2 in array "pop"

    ind2 = pop[max_sim_index]
    acc2 = acc[max_sim_index]
    p = rand()
    if p>0.5:
        return ind1, acc1, max_sim_index
    else:
        return ind2, acc2, max_sim_index

def conditional_choice(new_pop, new_Fit, altruism_indi, dim):
    #Calculate how many best solutions need to be intact
    num_pop_to_keep = round(new_pop.shape[0]-altruism_indi*2)
    print(f"the number of best solutions which is remained intact: {num_pop_to_keep}")

    # Sort in ascending order (lower fitness means better solution) => Check
    ind = np.argsort(new_Fit, axis=0)
    # print(ind, ind.shape)
    new_pop = new_pop[ind[:,0]]
    new_Fit = new_Fit[ind[:,0]]

    # Select the best (pop_size-altruism_indi) in the final population
    final_pop = new_pop[0:num_pop_to_keep,:]
    final_Fit = new_Fit[0:num_pop_to_keep,:]

    #Select 'altruism_indi' number of mediocre solutions from Woa alrogithm for the altruism operation (half of these will finally be selected)
    new_pop = new_pop[num_pop_to_keep:num_pop_to_keep+2*altruism_indi,:]
    new_Fit = new_Fit[num_pop_to_keep:num_pop_to_keep+2*altruism_indi,:]

    grouped_pop = np.zeros(shape=(altruism_indi,dim))
    grouped_fit = np.zeros(shape=(altruism_indi,1))
    count = 0
    while (len(new_pop)>0):
        grouped_pop[count], grouped_fit[count], pos2 = get_closest_ind(new_pop,new_Fit,beta=0.3)
        count+=1
        new_pop = np.delete(new_pop,[0,pos2],axis=0)     #check
        new_Fit = np.delete(new_Fit,[0,pos2],axis=0)     #check
    final_pop = np.concatenate((final_pop, grouped_pop), axis=0)
    final_Fit = np.concatenate((final_Fit, grouped_fit), axis=0)
    return final_pop, final_Fit