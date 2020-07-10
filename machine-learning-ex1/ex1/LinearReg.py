# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data= pd.read_csv("ex1data1.txt", header=None)

# Divide the dataset into X and y

X=data.iloc[:,:-1].values
y=data.iloc[:,1].values

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/5,random_state=0)
#Implement classifier based on Simple Linear Regression

from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression() 

simpleLinearRegression.fit(X,y)

y_predict=simpleLinearRegression.predict(X)
y_pro=simpleLinearRegression.predict([[3.5]])


















    
    