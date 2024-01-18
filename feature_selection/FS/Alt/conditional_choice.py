import numpy as np
from Py_FS.filter import SCC,PCC
from sklearn.preprocessing import LabelEncoder
from utils import TrainingClassifier
from numpy.random import rand

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
    # print(max(similarity_list))
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


