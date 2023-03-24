import numpy as np
import pandas as pd
import random

dataset = pd.read_csv("MV_Reg_noisy_data.csv")

X_train, X_test, y_train, y_test = train_test_split(data.drop('y', axis=1), data['y'], test_size=0.2, random_state=42)

shape = dataset.shape
x0 = []
for _ in range(shape[0]):
    x0.append(int(1))
x0 = pd.Series(x0)
dataset.insert(loc = 0, column = 'x0', value = x0)
# print(dataset.head)
# Assigning number of nodes in the hidden layer

hidden_nodes 6

# Assigning the weights
w_ji = [] # Input Layer to the hidden layer
for in range(hidden_nodes-1): 
    temp = []
    for in range(2):
        temp.append(random.randint(-1,1))
    w_ji.append(temp)
print("Original Weight 1 : ",w_ji)

w_kj = [] # Hidden Layer to the Output layer 
for in range(hidden_nodes):
    w_kj.append(random.randint(-1,1))
print("Original Weight 2 : ",w_kj)

#Defining the Sigmoid & Normalise function
def phi(x):
    return 1/(1+np.exp(-x))
def phi_dash(x):
    return phi(x)*(1-phi(x))
def func(x,a,b):
    return (a + (b-a)*x)

#Breaking the dataset into attributes and outputs for processing
x_features = np.array(dataset.iloc[:,:-1])
outputs np.array(dataset.iloc[:, -1])

a_min = np.min(outputs)
b_max= np.max(outputs)

#Starting the training process
epochs = 1000
Error = []
eat = 0.1

def summation(y_val,V,U,j,wkj,u,i,x_val):
    return_val = (y_val - V)*phi_dash(U)*wkj[j+1]*phi_dash(u[j])*x_val[i]
    return return_val

for itr in range(epochs):
    err = []
    for x_val, y_val in zip(x_features, outputs):
        unj [1,0,0,0,0,0]
        for j in range(1, hidden_nodes):
            for i in range(2):
                unj[j] += x_val[i]*w_ji[j-1][i]
        vnj = [1,0,0,0,0,0]
        for i in range(1,hidden_nodes):
            vnj[j] = phi(unj[j])
        Ukj = 0
        for j in range(hidden_nodes):
            Ukj += vnj[j]*w_kj[j]
        Vkj = 0
        Vkj = phi(Ukj)
        Vkj = func(Vkj,a_min,b_max)
        en = (0.5)*((y_val-Vkj)**2)
        err.append
        
        # BACKPROPAGATION ALGORITHM
        for j in range(hidden_nodes):
            w_kj[j] = w_kj[j] + eta*(y_val - Vkj)*phi_dash(Ukj)*vnj[j]
        for j in range(hidden_nodes-1):
            for i in range(2):
                temp = eta*summation(y_val,Vkj,Ukj,j,w_kj,unj,i,x_val)w_ji[j][i] = w_ji[j][i] + temp
    Error.append(np.mean(err))

print("Final Weight 1 : ",w_ji)
print("Final Weight 2: ",w_kj)
# print(Error)

import matplotlib.pyplot as plt
plt.title("Error Vs Iteration")
plt.plot(np.array(range(epochs)), np.array(Error),c='g')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show() 