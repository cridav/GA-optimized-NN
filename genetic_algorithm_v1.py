import numpy as np
import NN_mod3_GA as nn

"""
NN_mod3_GA : Contains the neural network
parameters_GA: dictionary with the weights for the neural network
W1 and b1 : Weights and biases input - hidden layer
W2 and b2 : Weights and bias hidden - output layer
parameters_GA = {   "W1": W1new,
                    "b1": b1new,
                    "W2": W2new,
                    "b2": b2new}                    
X: Input data
Y: Output labels
ini_pop : Population, every individual is a column
n_x : number of inputs
n_h : number of neurons in hidden layer
n_y : number of outputs
sorted_pop, pop : ranked population
parent_a, parent_b, child : vector of genes (indivudual extracted from population)

"""

def fitness(parameters_GA,X,Y):
    # Returns the fitness value of the evaluated individual
    A2, cache = nn.forward_propagation(X, parameters_GA)
    cost, error, numclas = nn.compute_cost(A2, Y, parameters_GA)
    fitness = numclas/A2.shape[1]
    return fitness
    
def vec_pop_to_weights(ini_pop, n_x, n_h, n_y,ind=0):
    # Translates the genes of the best individual in the population
    # into weights for the neural network
    #weights for hidden layer
    W1new = ini_pop[0:n_x*n_h,ind].reshape((n_h,n_x))
    b1new = ini_pop[(n_x*n_h):(n_x*n_h)+n_h,ind].reshape((n_h,1))
    #weights for output layer
    W2new = ini_pop[(n_x*n_h)+n_h:((n_x*n_h)+n_h)+n_h,ind].reshape((n_y,n_h))
    b2new = ini_pop[((n_x*n_h)+n_h)+n_h:((n_x*n_h)+n_h)+n_h+n_y,ind].reshape((1,n_y))    
    return W1new, b1new, W2new, b2new

def ind_to_weights(ini_pop, n_x, n_h, n_y):
    # Translates genes of the selected individual into weights
    W1new = ini_pop[0:n_x*n_h].reshape((n_h,n_x))
    b1new = ini_pop[(n_x*n_h):(n_x*n_h)+n_h].reshape((n_h,1))
    #weights for output layer
    W2new = ini_pop[(n_x*n_h)+n_h:((n_x*n_h)+n_h)+n_h].reshape((n_y,n_h))
    b2new = ini_pop[((n_x*n_h)+n_h)+n_h:((n_x*n_h)+n_h)+n_h+n_y].reshape((1,n_y))
    return W1new, b1new, W2new, b2new

def roulette(sorted_pop):
    # Select two individuals to be the parents, their probability of be selected is
    # proportional to their fitness value
    extractvector = sorted_pop[-1,:]
    fit=sorted_pop[-1,:]*100/sum(sorted_pop[-1,:])
    roulette=np.full(len(fit),100)
    roulette[0]=100-fit[0]
    #Create values proportional to the accuracy
    for i in range(len(fit)-1):
        roulette[i+1]=roulette[i]-fit[i+1]
    roulette[roulette<0]=0
    spin=np.random.randint(0,100,2)
    #Select first value pointed by the roulette
    index=np.where(roulette == roulette[roulette<=spin[0]][0])[0][0]
    parent_a = sorted_pop[:,index]
    
    #spin=np.random.randint(0,100)
    #     #Select first value pointed by the roulette
    index=np.where(roulette == roulette[roulette<=spin[1]][0])[0][0]
    parent_b = sorted_pop[:,index]
    return parent_a, parent_b

def select_best(pop, num_parents=1):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((pop.shape[0],num_parents))
    for i in range(num_parents):
        parents[:,i] = pop[:,i]
    return parents

def crossover(parent_a, parent_b):
    #exclude the fitness value in vector and mix the genes of the parents into a child
    a=len(parent_a)
    child = np.zeros(a)
    child[0:a//2]=parent_a[0:a//2]
    child[a//2:a]=parent_b[a//2:a]
    return child

def mutation(child):
    # Mutation changes a single random gene in every children
    mut_rate = np.random.uniform(-1,1,1)
    child_mut = child
    child_mut[np.random.randint(len(child))] = child[np.random.randint(len(child))]*mut_rate
    return child_mut
