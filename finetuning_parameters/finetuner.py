#[2016]-"The whale optimization algorithm"]

import numpy as np
import random
from numpy.random import rand
from finetuning_parameters.utils import *

def init_position(lb, ub, N, dim):
    """
    Initialize the position of Whales with:
    N: number of population of Whales in the Algorithm
    lb: Lower bound
    ub: Upper bound
    dim: number of bits
    """
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):        
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x

def multiple_gene_crossover(parent1, parent2, num_genes):
    """
    Perform multiple gene crossover between two parent individuals.

    Args:
    parent1 (list): The genetic information of the first parent.
    parent2 (list): The genetic information of the second parent.
    num_genes (int): The number of genes to exchange between parents.

    Returns:
    child1 (list): Genetic information of the first child.
    child2 (list): Genetic information of the second child.
    """

    # # Ensure both parents have the same length
    if len(parent1) != len(parent2):
        raise ValueError("Parent chromosomes must have the same length.")

    # Random one dimension from 1 to dim
    index   = np.random.randint(low = 1, high = len(parent1)-num_genes) # why
    # Crossover
    # if index > round(len(parent1/2)):
    child1 = np.concatenate((parent1[0:index] , parent2[index:index+num_genes], parent1[index+num_genes:]))
    #     # child2 = np.concatenate((parent2[0:index] , parent1[index:]))
    # else:
    #       child1 = np.concatenate((parent2[0:index] , parent1[index:]))
    
    return child1

def roulette_wheel(prob):
    num = len(prob)
    C   = np.cumsum(prob)
    P   = rand()
    for i in range(num):
        if C[i] > P:
            index = i
            break
        else: 
            index = random.randint(0, num+1)
    
    return index

def finetuner(xtrain, ytrain, opts):
    print("Welcome to parameters finetuner!")
    # parameters
    ub      = 1
    lb      = 0
    thres   = 0.5
    b       = 1 # constant
    CR      = 0.7     # crossover rate
    MR      = 0.2    # mutation rate

    N           = opts['N'] # number of Whales population
    max_iter    = opts['T'] # iterations
    if 'b' in opts:
        b       = opts['b']
    if 'CR' in opts:
        CR   = opts['CR'] 
    if 'MR' in opts: 
        MR   = opts['MR']  
    
    # Dimention
    dim = 42       # number of bits which serve for quantization
    if np.size(lb)==1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    
    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Calculate fitness of each search agent at first iteration
    fit     = np.zeros([N, 1], dtype= 'float')
    Xgb     = np.zeros([1, dim], dtype= 'float')
    fitG    = float('inf')

    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        print(f"Fit at the first iteration: {fit[i,0]}")
        if fit[i,0] < fitG:
            Xgb[0,:]    = X[i,:]    # Choose the best whale in the population
            fitG        = fit[i,0]  # Store the best fit in the population

    # Divide the population into 3 groups including Exploitative sub-population, Exploratory sub-population, hesitate sub-population at the first iteration
    v = np.max(fit) - np.min(fit)
    Xexplore    = X[fit[:,0]>(np.max(fit)-v/3),:]                                                       # shape of Xexplore array is [k, dim] with k is the number of individuals in the Exploratory sub-population 
    fitExplore  = fit[fit[:,0]>(np.max(fit)-v/3),:]                                                     # shape of fitExplore array is [k, 1] with k is the number of individuals in the Exploratory sub-population 
    K_explore   = len(Xexplore)
    Xexploit    = X[fit[:,0]<(np.min(fit)+v/3),:]                                                       # shape of Xexploit array is [k, dim] with k is the number of individuals in the Exploitative sub-population
    fitExploit  = fit[fit[:,0]<(np.min(fit)+v/3),:]                                                     # shape of fitExploit array is [k, 1] with k is the number of individuals in the Exploitative sub-population 
    K_exploit   = len(Xexploit)
    Xhesitate   = np.array([i for i in X if ((i not in Xexplore) and (i not in Xexploit) )])            # shape of Xhesitate array is [k, dim] with k is the number of individuals in the hesitate sub-population
    fitHesitate = np.array([i for i in fit if ((i not in fitExplore) and (i not in fitExploit) )])      # shape of fitHesitate array is [k, 1] with k is the number of individuals in the hesitate sub-population
    K_hesitate  = len(Xhesitate)

    # Set up curve
    curve   = np.zeros([1, max_iter], dtype='float')
    t       = 0

    curve[0,t] = fitG.copy()
    print("Generation:", t + 1)
    print("Best (WOA):", curve[0,t])
    t += 1

    while t < max_iter:
        # Define a, linearly decreases from 2 to 0 
        a = 2 - t * (2 / max_iter)
        
        # Processing for hestitate sub-population
        for i in range(K_hesitate):
            # Parameter A (2.3)
            A = 2 * a * rand() - a
            # Paramater C (2.4)
            C = 2 * rand()
            # Parameter p, random number in [0,1]
            p = rand()
            # Parameter l, random number in [-1,1]
            l = -1 + 2 * rand()  
            # Whale position update (2.6)
            if p  < 0.5:
                # {1} Encircling prey
                if abs(A) < 1:
                    for d in range(dim):
                        # Compute D (2.1)
                        Dx     = abs(C * Xgb[0,d] - Xhesitate[i,d])
                        # Position update (2.2)
                        Xhesitate[i,d] = Xgb[0,d] - A * Dx
                        # Boundary
                        Xhesitate[i,d] = boundary(Xhesitate[i,d], lb[0,d], ub[0,d])
                
                # {2} Search for prey
                elif abs(A) >= 1:
                    for d in range(dim):
                        # Select a random whale
                        k      = np.random.randint(low = 0, high = K_hesitate)
                        # Compute D (2.7)
                        Dx     = abs(C * Xhesitate[k,d] - Xhesitate[i,d])
                        # Position update (2.8)
                        Xhesitate[i,d] = Xhesitate[k,d] - A * Dx
                        # Boundary
                        Xhesitate[i,d] = boundary(Xhesitate[i,d], lb[0,d], ub[0,d])
            
            # {3} Bubble-net attacking 
            elif p >= 0.5:
                for d in range(dim):
                    # Distance of whale to prey
                    dist   = abs(Xgb[0,d] - Xhesitate[i,d])
                    # Position update (2.5)
                    Xhesitate[i,d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0,d] 
                    # Boundary
                    Xhesitate[i,d] = boundary(Xhesitate[i,d], lb[0,d], ub[0,d])
        
        # Duplicate Xexploit and Xexplore
        copy_Xexploit = Xexploit.copy()
        copy_Xexplore = Xexplore.copy()

        # Processing for Exploitative sub-population
        for i in range(K_exploit):
            # Parameter A (2.3)
            A = 2 * a * rand() - a
            # Paramater C (2.4)
            C = 2 * rand()
            # Parameter p, random number in [0,1]
            p = rand()
            # Parameter l, random number in [-1,1]
            l = -1 + 2 * rand()  
            # {3} Bubble-net attacking 
            if p<0.5:
                for d in range(dim):
                    # Compute D (2.1)
                    Dx     = abs(C * Xgb[0,d] - Xexploit[i,d])
                    # Position update (2.2)
                    Xexploit[i,d] = Xgb[0,d] - A * Dx
                    # Boundary
                    Xexploit[i,d] = boundary(Xexploit[i,d], lb[0,d], ub[0,d])
            else:
                for d in range(dim):
                    # Distance of whale to prey
                    dist   = abs(Xgb[0,d] - Xexploit[i,d])
                    # Position update (2.5)
                    Xexploit[i,d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0,d] 
                    # Boundary
                    Xexploit[i,d] = boundary(Xexploit[i,d], lb[0,d], ub[0,d])

        # Processing for Exploratory sub-population
        for i in range(K_explore):
            # Parameter A (2.3)
            A = 2 * a * rand() - a
            # Paramater C (2.4)
            C = 2 * rand()
            # Parameter p, random number in [0,1]
            p = rand()
            # Parameter l, random number in [-1,1]
            l = -1 + 2 * rand()  
            # Whale position update (2.6)             
            # {2} Search for prey
            for d in range(dim):
                # Select a random whale
                k      = np.random.randint(low = 0, high = K_explore)
                # Compute D (2.7)
                Dx     = abs(C * Xexplore[k,d] - Xexplore[i,d])
                # Position update (2.8)
                Xexplore[i,d] = Xexplore[k,d] - A * Dx
                # Boundary
                Xexplore[i,d] = boundary(Xexplore[i,d], lb[0,d], ub[0,d])

        # Processing for Genetics algorithm
        # Probability
        inv_fitExplore = 1 / (1 + fitExplore)
        probExplore    = inv_fitExplore / np.sum(inv_fitExplore)
        inv_fitExploit = 1 / (1 + fitExploit)
        probExploit    = inv_fitExploit / np.sum(inv_fitExploit)

        # Number of crossovers
        Nc = 0
        for i in range(N):
            if rand() < CR:
                Nc += 1
        
        X1 = np.zeros([Nc, dim], dtype='float64')
        for i in range(Nc):
            # Parent selection
            k1      = roulette_wheel(probExplore)
            k2      = roulette_wheel(probExploit)
            P1      = copy_Xexplore[k1,:].copy()    # Parent 1
            P2      = copy_Xexploit[k2,:].copy()    # Parent 2
            # Crossover
            X1[i,:] = multiple_gene_crossover(P1, P2, round(random.uniform(0.1,0.45)*dim/2))
            # Mutation
            for d in range(dim):
                if rand() < MR:
                    X1[i,d] = 1 - X1[i,d]

            
        # Binary conversion
        print(f" Shapes of Xexplore, Xexploit, Xhesitate, X1 is {Xexplore.shape, Xexploit.shape, Xhesitate.shape, X1.shape} respectively")
        if Xhesitate.shape[0]!=0:
            X = np.concatenate((Xexplore, Xexploit, Xhesitate), axis=0)
        else:
            X = np.concatenate((Xexplore, Xexploit), axis=0)
        Xbin = binary_conversion(X, thres, N, dim)
        Xbinnew = binary_conversion(X1, thres, Nc, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
        # print(f"WOA Fitness value at a {t} interation: {fit}")
        # Fitness for genetic part
        fit_new = np.zeros([Nc, 1], dtype='float')
        for i in range(Nc):
            fit_new[i,0] = Fun(xtrain, ytrain, Xbinnew[i,:], opts)
            if fit_new[i,0] < fitG:
                Xgb[0,:] = X1[i,:]
                fitG     = fit_new[i,0]
        # print(f"GA Fitness value at a {t} interation: {fit_new}")

        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print(f"Best individual: {Xgb[0,:]}")
        print("Best (WOA_GA):", curve[0,t])
        t += 1 

        # Elitism 
        XX_bin = np.concatenate((Xbin , Xbinnew), axis=0)
        _, index = np.unique(XX_bin, return_index=True, axis=0)
        XX      = np.concatenate((X , X1), axis=0)[index]
        FF      = np.concatenate((fit , fit_new), axis=0)[index]
        XX = XX[FF[:,0]<1,:]
        FF = FF[FF[:,0]<1,:]
        altruism_indi = XX.shape[0]-N
        if altruism_indi>0:
            # Sort in ascending order
            # print(f"Length of Fit function before sorting: {len(FF)}")
            # ind = np.argsort(FF, axis=0)
            # for i in range(N):
            #     X[i,:]   = XX[ind[i,0],:]
            #     fit[i,0] = FF[ind[i,0]]   
            X, fit = conditional_choice(XX, FF, altruism_indi, dim)

        else: 
            print("==========Regenerate population=============")
            # Initialize position
            extra_N = N - XX.shape[0] 
            extra_X = init_position(lb, ub, extra_N, dim)

            # Binary conversion
            extra_Xbin = binary_conversion(extra_X, thres, extra_N, dim)
            X = np.concatenate((XX, extra_Xbin), axis=0)
        
        # Divide the population into 3 groups including Exploitative sub-population, Exploratory sub-population, hesitate sub-population at the maintaining iteration
        v = np.max(fit) - np.min(fit)
        Xexplore    = X[fit[:,0]>(np.max(fit)-v/3),:]                                                       # shape of Xexplore array is [k, dim] with k is the number of individuals in the Exploratory sub-population 
        fitExplore  = fit[fit[:,0]>(np.max(fit)-v/3),:]                                                     # shape of fitExplore array is [k, 1] with k is the number of individuals in the Exploratory sub-population 
        K_explore   = len(Xexplore)
        Xexploit    = X[fit[:,0]<(np.min(fit)+v/3),:]                                                       # shape of Xexploit array is [k, dim] with k is the number of individuals in the Exploitative sub-population
        fitExploit  = fit[fit[:,0]<(np.min(fit)+v/3),:]                                                     # shape of fitExploit array is [k, 1] with k is the number of individuals in the Exploitative sub-population 
        K_exploit   = len(Xexploit)
        hesitate_list = []
        for i in range(X.shape[0]):
            if i < fit.shape[0]:
                if (fit[i] not in fitExploit) and (fit[i] not in fitExplore):
                    hesitate_list.append(X[i,:])
                else:
                    continue
            else:
                hesitate_list.append(X[i,:])
        Xhesitate = np.array(hesitate_list)
        K_hesitate  = len(Xhesitate)
        if K_hesitate==0:
            ind = np.argsort(fitExploit, axis=0)
            Xhesitate = Xexploit[ind[round(K_exploit/2):,0],:]
            K_hesitate  = len(Xhesitate)
            Xexploit = Xexploit[ind[0:round(K_exploit/2),0],:]
            fitExploit = fitExploit[ind[0:round(K_exploit/2),0],:]
            K_exploit   = len(Xexploit)
        # print(f" length of Xexplore, Xexploit, Xhesitate is {K_explore, K_exploit, K_hesitate} respectively")

    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)    
    iterations               = dequantize_from_binary_array(Gbin[0:6], 50, 690)        #6bits
    learning_rate           = dequantize_from_binary_array(Gbin[6:20], 0.001, 0.5)      #14bits
    depth                   = dequantize_from_binary_array(Gbin[20:23], 2, 9)           #3bits
    l2_leaf_reg             = dequantize_from_binary_array(Gbin[23:26], 2, 9)           #3bits
    random_strength         = dequantize_from_binary_array(Gbin[26:34], 1e-8, 10)       #8bits
    bagging_temperature     = dequantize_from_binary_array(Gbin[34:42], 1e-8, 10)       #8bits
    params = {"iterations": round(iterations), "learning_rate": learning_rate, "depth": round(depth),
              "l2_leaf_reg": l2_leaf_reg, "random_strength": random_strength, "bagging_temperature": bagging_temperature}
    # Create dictionary
    woa_data = {'params': params, 'c': curve}
    
    return woa_data 
