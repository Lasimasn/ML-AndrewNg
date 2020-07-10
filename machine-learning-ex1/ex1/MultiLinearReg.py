# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 00:33:41 2020

@author: LasimaSN
"""


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data= pd.read_csv("ex1data2.txt", header=None)
X=data.iloc[:,[0,1]]
y=data.iloc[:,2] 

bedroom=pd.get_dummies(X[1],drop_first=True)

X=X.drop(1,axis=1)

X=pd.concat([X,bedroom],axis=1)

from sklearn.linear_model import LinearRegression
linReg= LinearRegression()
linReg.fit(X,y)