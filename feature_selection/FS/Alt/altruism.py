import numpy as np
from Py_FS.filter import SCC,PCC
from sklearn.preprocessing import LabelEncoder
from utils import TrainingClassifier

params = {"iterations": round(350), "learning_rate": 0.135, "depth": round(6),
              "l2_leaf_reg": 1, "random_strength": 0.5, "bagging_temperature": 0.5}

def generate_scc(train_data,train_label):
    le = LabelEncoder()
    train_label = le.fit_transform(train_label)
    classifier = TrainingClassifier(params)
    (train_data,train_label) = classifier.encode_feature((train_data,train_label))
    scc = SCC(train_data,train_label)
    scc.run()
    return scc.feature_scores

def generate_pcc(train_data,train_label):
    le = LabelEncoder()
    train_label = le.fit_transform(train_label)
    classifier = TrainingClassifier(params)
    (train_data,train_label) = classifier.encode_feature((train_data,train_label))
    pcc = PCC(train_data,train_label)
    pcc.run()
    return pcc.feature_scores

def hamming_distance(b1,b2):
    ans = 0
    for i in range(len(b1)):
        ans += not(b1[i]==b2[i])
    return ans

def similarity(beta, chromosome1, chromosome2, acc1, acc2):
    H_d = hamming_distance(chromosome1,chromosome2)
    D_a = abs(acc1-acc2)
    if (H_d !=0):
        if (D_a !=0): 
            S = beta/H_d + (1-beta)/D_a
        else :
            S = 99999
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
    return ind1, ind2, acc1, acc2, max_sim_index

def group_population(pop, acc):
    grouped_pop = np.zeros(shape=pop.shape)
    grouped_fit = np.zeros(shape=acc.shape)
    count = 0
    while (len(pop)>0):
        grouped_pop[count], grouped_pop[count+1], grouped_fit[count], grouped_fit[count+1], pos2 = get_closest_ind(pop,acc,beta=0.3)
        count+=2
        pop = np.delete(pop,[0,pos2],axis=0)     #check
        acc = np.delete(acc,[0,pos2],axis=0)     #check
    return grouped_pop, grouped_fit

def Altruism(new_pop, new_Fit, scc_score, pcc_score, altruism_indi, pop_size, alpha = 0.5):
    """
    new_pop: The whole population out of which half will be selected to be the population of next generation (of size 2*pop_size)
    altruism_indi: Number of individuals that will be sacrificed (altruism)
    pop_size = The original size of the population that will be used in the next generation
    alpha, beta: weights for asserting the dominance of one individual over another

    returns: final_pop- a population of size = 'pop_size', where 'altruism_indi' number of mediocre solutions were sacrificed
    """    
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

    # Group population according to similarity indices   
    grouped_pop, grouped_Fit = group_population(new_pop, new_Fit)

    #Initialize the population who will finally survive the altruism operation
    altruism_pop = np.zeros(shape=(altruism_indi, new_pop.shape[1]))
    altruism_Fit = np.zeros(shape=(altruism_indi, 1))
    count = 0
    while (count/2)<altruism_indi:
        player1 = grouped_pop[count]
        player1_fit = grouped_Fit[count]
        player2 = grouped_pop[count+1]
        player2_fit = grouped_Fit[count+1]

        idx1 = np.where(player1==1)[0]
        idx2 = np.where(player2==1)[0]
        scc1 = np.average(scc_score[idx1])
        scc2 = np.average(scc_score[idx2])
        pcc1 = np.average(pcc_score[idx1])
        pcc2 = np.average(pcc_score[idx2])
        
        #Compute which candidate soln has more potential for reaching global optima (Check the description in our paper)
        player1_score = alpha*scc1 + (1-alpha)*pcc1
        player2_score = alpha*scc2 + (1-alpha)*pcc2

        if player1_score <= player2_score:
            altruism_pop[int(count/2)] = player1
            altruism_Fit[int(count/2)] = player1_fit
        else:
            altruism_pop[int(count/2)] = player2
            altruism_Fit[int(count/2)] = player2_fit

        count+=2

    # Merge the population that was kept intact and the altruistic individuals to form the final population
    final_pop = np.concatenate((final_pop, altruism_pop), axis=0)
    final_Fit = np.concatenate((final_Fit, altruism_Fit), axis=0)

    return final_pop.astype(int), final_Fit


