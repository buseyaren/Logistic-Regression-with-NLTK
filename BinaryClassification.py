# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:46:06 2019

@author: tekin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#eğitim boyutu
TRAIN_SIZE = 0.75
# parametreler
FILTER_STEM = True
TRAIN_PORTION = 0.8
RANDOM_STATE = 7

#veri setindeki verilerin çekilmesi
df = pd.read_csv('Data/sentiment140.csv',
                 encoding="ISO-8859-1",
                 names=["target", "ids", "date", "flag", "user", "text"])
#Kategoriler
decode_map = {0: -1, 2: 0, 4: 1}
df.target = df.target.apply(lambda x: decode_map[x])
#input features
X=df[['ids','date','flag','user','text']]
#output, target
Y=df[['target']]

sns.lmplot('ids','text',df,hue='target',fit_reg=False)
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()
#veri seti bilgilendirmesi
(p, n) = X.shape
theta = np.zeros((n+1,1)) # intializing theta with all zeros
ones = np.ones((p,1))
#here we are appending one column with values of ones
X = np.hstack((ones, X))
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
z = np.dot(X, theta)
h = sigmoid(z)
