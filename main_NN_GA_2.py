import genetic_algorithm_v1 as g
import NN_mod3_GA as nn

import numpy as np
import matplotlib.pyplot as plt

# lOAD DATA FROM NN_MOD3_GA.PY
X, Y, XTest, YTest = nn.load_data()
#parameters, costit, errorit, numclas = nn.nn_model(X, Y, 7, 200, True, 1.2)
#np.random.seed(9001)
pop_size = 100
generations = 300
#mutation = 0.1

#Get size of the NN
n_x, n_h, n_y = nn.layer_sizes(X,Y)

#-=-=-=-=-=     CREATE INITIAL POPULATION
size=(n_x*n_h)+n_h+(n_h*n_y)+n_y
ini_pop=np.random.uniform(low=-1, high=1,size=(size,pop_size))

accu = np.zeros(generations)

#-=-=-=-=-=     RUN GENERATIONS
for generation in range(generations):
    #-=-=-=-=-=     EVALUATE FITNESS
    fitness=np.zeros((1,pop_size))
    for ind in range(pop_size):
        W1new, b1new, W2new, b2new = g.vec_pop_to_weights(ini_pop,n_x,n_h,n_y,ind)
        parameters_GA = {   "W1": W1new,
                            "b1": b1new,
                            "W2": W2new,
                            "b2": b2new}
        # Evaluate all the population
        fitness[0,ind]=g.fitness(parameters_GA, X, Y)

    #Append fitness to the population
    #Flip array, best individuals at first positions
    pop_fitness=np.append(ini_pop,fitness,axis=0)
    #make a ranking, best individuals first
    sorted_pop=pop_fitness[:,pop_fitness[size,:].argsort()]
    temp_arr=sorted_pop.copy()
    for i in range(pop_size):
        sorted_pop[:,i]=temp_arr[:,pop_size-i-1]
    
    
    # PRINT (VISUALIZE BEST 5 INDIVIDUALS SO FAR)
    print('Generation: ',generation)
    vis=5
    for it in range(vis):
        W1vis, b1vis, W2vis, b2vis = g.ind_to_weights(sorted_pop[0:size,it],n_x,n_h,n_y)
        parameters_GA = {   "W1": W1vis,
                        "b1": b1vis,
                        "W2": W2vis,
                        "b2": b2vis}

        TN, FP, FN, TP = nn.plot_confusion_matrix(X, Y, parameters_GA,False)
        
        print('Ind:', it, 'acc: ',round(sorted_pop[-1,it],2),'FN:', FN, 'FP:', FP, 'TN:', TN, 'TP:', TP)
        # END_PRINT
    print("-"*10)
    
    accu[generation] = sorted_pop[-1,0]
    #-=-=-=-=-=     SELECT - ROULETTE
    #-=-=-=-=-=     CROSSOVER AND MUTATION
    #Apply elitism, the best one goes to the next generation
    ini_pop[:,0] = sorted_pop[0:size,0]
    #generate population, N-1
    for N in range(pop_size-1):
    #for N in range(pop_size): 
        parent_a, parent_b = g.roulette(sorted_pop)
        child = g.crossover(parent_a[0:size], parent_b[0:size])
        ini_pop[:,N+1]=g.mutation(child)
        #ini_pop[:,N]=g.mutation(child)

#-=-=-=-=-=         END OF ITERATIONS

#EVALUATE BEST INDIVIDUAL
best = g.select_best(ini_pop)

W1new, b1new, W2new, b2new = g.ind_to_weights(best,n_x,n_h,n_y)
parameters_GA = {   "W1": W1new,
                    "b1": b1new,
                    "W2": W2new,
                    "b2": b2new}


#Training data
TN, FP, FN, TP = nn.plot_confusion_matrix(X, Y, parameters_GA)
plt.figure()
plt.plot(accu*100)
plt.title("Training\nAccuracy vs generation")
plt.xlabel("Generation")
plt.ylabel("Accuracy [%]")
accuracy = (TP + TN)/(TN+FP+FN+TP)
sensitivity = TP/(TP+FN)
especificity = TN/(FP+TN)
precision = TP/(TP+FP)
print('Accuracy: ', round(accuracy*100,2),' %')
print('Sensitivity: ', round(sensitivity*100,2),' %')
print('Especificity: ', round(especificity*100,2),' %')
print('Precision: ', round(precision*100,2),' %')
print('-'*10)

#Validation data
TN, FP, FN, TP = nn.plot_confusion_matrix(XTest, YTest, parameters_GA)
accuracy = (TP + TN)/(TN+FP+FN+TP)
sensitivity = TP/(TP+FN)
especificity = TN/(FP+TN)
precision = TP/(TP+FP)
print('Accuracy: ', round(accuracy*100,2),' %')
print('Sensitivity: ', round(sensitivity*100,2),' %')
print('Especificity: ', round(especificity*100,2),' %')
print('Precision: ', round(precision*100,2),' %')
plt.show()