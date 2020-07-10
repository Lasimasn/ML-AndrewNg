# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data= pd.read_csv('ex1data1.txt', header=None)

# Divide the dataset into X and y

X=data.iloc[:,0].values # Read first column
y=data.iloc[:,1].values # Read second column
m=len(X)

plt.scatter(X,y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000')
plt.savefig('graph1.png')
plt.show()

X=X[:,np.newaxis]
y=y[:,np.newaxis]
# adds another column to the data to accomodate intercept term ie theta
theta=np.zeros([2,1])
#theta is the zero array
iterations = 1500 
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X))


def computeCost(X,y,theta):
    h_0x =np.dot(X, theta) - y
    return np.sum(h_0x**2) / (2*m)
J=computeCost(X, y, theta)
print (J)
      

def gradDescent(X,y,theta,iterations,alpha):
    
    for i in range(iterations):
        
        h_0x= np.dot(X,theta) - y
        temp= np.dot(X.T, h_0x)
        theta = theta - (alpha/m) * temp
    return theta

theta=gradDescent(X,y,theta,iterations,alpha)
print (theta)

J=computeCost(X, y, theta)
print(J)

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X,theta))
plt.savefig('graph2.png')
plt.show()




















    
    