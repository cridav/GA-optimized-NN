"""
Created on Fri Jan 11 21:36:25 2019

@author: cristiam
"""
# This version will have hard limiter for Y, both activation functions

import itertools
# import libraries
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
    if os.name == 'nt':
        path = ""
    else:
        path = "/home/cristiam/Documents/ISI_WEiTI/II/EINIS/project"
        # Set working directory and load data
        os.chdir(path)
        
    Train = pd.read_csv('training_data.csv', sep='\s*,\s*',
                        header=0, encoding='ascii', engine='python')
    # Sort the dataframe by columns, so that we will get the biggest amount of '1'
    # for the training dataset
    Train = Train.sort_values(by='Resp', ascending=False)

    # Obtain set of characters belonging to the sequences
    # RTSeq contains the complete sequences in both data sets
    char_set = list(set("".join(Train['RTSeq'])))
    char_set_size = len(char_set)
    # Assign a number depending on the number of different letters found
    # in the sequence

    num = dict(zip(char_set, range(1, char_set_size + 1)))
    # Replace every letter with their new numerical values
    new = Train.at[0, 'PRSeq']
    # Covert colum 'RtSeq' into list
    Trainlist = Train['RTSeq'].tolist()
    # size of the longest element in the list
    col = len(max(Trainlist, key=len))
    # size of the list
    row = len(Trainlist)
    # create matrix with specified size
    rts = np.zeros((row, col))
    # CREATE ANOTHER VECTOR WITH THE SUM
    # OF EVERY CHARACTER IN THE SEQUENCE
    # IN COLUMN A AND THE NUMBER OF CHARACTERS
    # IN COLUMN B
    sumrts = np.zeros((row, 2))
    sumrts[:, 0] = rts.sum(axis=1)
    sumrts[:, 1] = np.count_nonzero(rts, axis=1)
    # turn letters into numbers, empty spaces with cero
    for i in range(0, row):
        rts[i][0:len(Trainlist[i])] = list(map(num.get, Trainlist[i]))

    # CREATE ANOTHER VECTOR WITH THE SUM
    # OF EVERY CHARACTER IN THE SEQUENCE
    # IN COLUMN A AND THE NUMBER OF CHARACTERS
    # IN COLUMN B
    sumrts = np.zeros((row, 2))
    sumrts[:, 0] = rts.sum(axis=1)
    sumrts[:, 1] = np.count_nonzero(rts, axis=1)
    # Covert colum 'PRSeq' into list
    Trainlist = Train['PRSeq'].tolist()
    incoms = Trainlist
    incoms = [incom for incom in incoms if str(incom) != 'nan']
    # size of the longest element in the list
    col = len(max(incoms, key=len))
    # size of the list
    row = len(Trainlist)
    # create matrix with specified size
    prs = np.zeros((row, col))

    # turn letters into numbers, empty spaces with cero
    for i in range(0, len(incoms)):
        prs[i][0:len(incoms[i])] = list(map(num.get, incoms[i]))
    # CREATE ANOTHER VECTOR WITH THE SUM
    # OF EVERY CHARACTER IN THE SEQUENCE
    # IN COLUMN A AND THE NUMBER OF CHARACTERS
    # IN COLUMN B
    sumprs = np.zeros((row, 2))
    sumprs[:, 0] = prs.sum(axis=1)
    sumprs[:, 1] = np.count_nonzero(prs, axis=1)

    # Another input column from Train
    C = Train['VLt0'].tolist()
    D = Train['CD4t0'].tolist()
    seq = np.column_stack((C, D))
    seq = np.transpose(seq)
    # CREATE VECTORS WITH THE NEW IDEA OF COUNT CHARACTERS AND SUM THEM
    A = np.transpose(sumprs)
    B = np.transpose(sumrts)

    # A=np.transpose(prs)
    # B=np.transpose(rts)
    seq = np.concatenate((A, B, seq))
    X = seq

    C = Train['Resp'].tolist()
    Y = C
    # Y=np.transpose(C)
    Y = np.multiply(1, Y)
    Y = np.array(Y)[np.newaxis]

    lrate = 1.2

    XX = X
    YY = Y

    # X=X[:,400:800]
    # Y=Y[:,400:800]
    #X = X[:, 0:400]
    #Y = Y[:, 0:400]
    a=150
    X = np.concatenate((X[:, 0:a],X[:, 206:206+a]),axis=1)
    Y = np.concatenate((Y[:, 0:a],Y[:, 206:206+a]),axis=1)
    XTest = np.concatenate((XX[:, a+1:206],XX[:, 206+a:206+a+(206-a+1)]),axis=1)
    YTest = np.concatenate((YY[:, a+1:206],YY[:, 206+a:206+a+(206-a+1)]),axis=1)

    #XTest = XX[:, 400:600]
    #YTest = YY[:, 400:600]

    # Apply logarithm to the biggest values, so that data will be between 0 and 10
    X[0:4, :] = np.log10(X[0:4, :])
    XTest[0:4, :] = np.log10(XTest[0:4, :])
    # Add 1 in order to avoid inf values
    X[5, :] = 1 + X[5, :]
    X[5, :] = np.log10(X[5, :])
    XTest[5, :] = 1 + XTest[5, :]
    XTest[5, :] = np.log10(XTest[5, :])

    # X=X[1779:1781,:]
    experiment = []

    return X, Y, XTest, YTest

    
    
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # Set a dictionary
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def layer_sizes(X, Y):
    n_x = X.shape[0]  # Input layer
    n_h = 7  # Hidden layer
    n_y = Y.shape[0]  # Output layer
    return n_x, n_h, n_y


def forward_propagation(X, parameters):
    # evaluate input data with current neural network (forward process)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Wa+b
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    rA2 = np.absolute(np.rint(A2))
    error = Y - rA2
    numclas = np.count_nonzero(error == 0)
    error = 1 - np.count_nonzero(error == 0) / error.shape[1]
    logprobs = np.where(A2 != 0, np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2)), 0)
    cost = - np.sum(logprobs) / m
    return cost, error, numclas


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    # Activation function is sigmoidal
    # dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 1))
    # dZ1 = np.multiply(dZ1,A1)
    # End sigmoidal
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations, print_cost, lrate):
    np.random.seed(3)
    #n_x = layer_sizes(X, Y)[0]
    #n_y = layer_sizes(X, Y)[2]
    n_x, _, n_y = layer_sizes(X,Y)
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    print(num_iterations)
    costit = []
    errorit = []
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # i=0
        # while True:
        A2, cache = forward_propagation(X, parameters)
        cost, error, numclas = compute_cost(A2, Y, parameters)
        
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, lrate)
        
        if i % 100 == 0:
            print("Epoch %i, layers: %i, cost: %f, clas. error: %f, classified: %i out of %i, accuracy: %.2f" % (
                i, n_h, cost, error, numclas, Y.shape[1], numclas / Y.shape[1] * 100))
            costit.append(cost)
            errorit.append(error)
        if cost < 0.4:
            break
        i += 1
    return parameters, costit, errorit, numclas

def validate(parameters, costit, errorit, numclas, X, Y, XTest, YTest):
    plt.figure()
    plt.plot(costit)
    plt.title('Cost Vs epoch')
    plt.figure()
    plots(parameters, X, Y)
    plt.title("Training accuracy: %i out of %i" % (numclas, Y.shape[1]), fontsize=8)
    plt.title("Training accuracy: {} out of {}, accuracy: {} %".format(numclas, Y.shape[1],numclas*100/Y.shape[1]), fontsize=8)

    plt.figure()
    A2, _ =plots(parameters, XTest, YTest)
    cost, error, numclas = compute_cost(A2, YTest, parameters)
    val_acc = numclas
    print(val_acc)
    #plt.title("Validation accuracy: %i out of %i, accuracy: %d %" % (val_acc, YTest.shape[1],val_acc*100/YTest.shape[1]), fontsize=8)
    plt.title("Validation accuracy: {} out of {}, accuracy: {} %".format(val_acc, YTest.shape[1],val_acc*100/YTest.shape[1]), fontsize=8)

    #print("Validation accuracy: {} out of {}, accuracy: {} %".format(val_acc, YTest.shape[1],val_acc*100/YTest.shape[1]))


def plots(parameters, seqs, labels):
    x = np.linspace(1, labels.shape[1], labels.shape[1])
    plt.plot(x, np.transpose(labels))

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Implement Forward Propagation to calculate A2 (probability)
    # Wa+b
    Z1 = np.dot(W1, seqs) + b1
    A1 = np.tanh(Z1)  # tanh activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid activation function

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    plt.scatter(x, np.around(np.transpose(A2)), s=2)
    plt.plot(x, np.around(np.transpose(A2)), linewidth=0.5)
    return A2, cache




def plot_confusion_matrix(plot_test_seqs, plot_test_labels, parameters,plott=True):
    test_predict, cache = forward_propagation(plot_test_seqs, parameters)
    test_predict = np.around(test_predict, decimals=0)
    test_predict = test_predict.astype(int)
    cnf_matrix = confusion_matrix(y_true=plot_test_labels.flatten(), y_pred=test_predict.flatten())
    if plott == True:
        #print(cnf_matrix)
        classes = ["Negative", "Positive"]

        plt.figure()
        plt.imshow(cnf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = cnf_matrix.max() / 2.0
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(
                j,
                i,
                cnf_matrix[i, j],
                horizontalalignment="center",
                color="white" if cnf_matrix[i, j] > thresh else "black",
            )

        plt.ylabel("True label", fontsize=16)
        plt.xlabel("Predicted label", fontsize=16)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        # plt.close()
        #plt.show()

    return cnf_matrix[0,0],cnf_matrix[0,1],cnf_matrix[1,0],cnf_matrix[1,1]



if __name__ == '__main__':
    path = './'
    X, Y, XTest, YTest = load_data()
    parameters, costit, errorit, numclas = nn_model(X, Y, 7, 200, True, 1.2)
   
    
    validate(parameters, costit, errorit, numclas,X,Y, XTest, YTest)
    plot_confusion_matrix(X, Y, parameters)
    plt.show()
    #G.priiint()
    
    