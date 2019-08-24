# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:13:27 2019

@author: robin
"""
import pandas as pd
import numpy as np
st=[]
#np.set_printoptions(suppress=True)
data=pd.read_csv("dota.csv")
#y=np.asanyarray(data[['embedding']])
#x=np.asfarray(y,float)
#data=np.loadtxt('dota1.csv',delimiter=',',skiprows=1,dtype=float)
#data=np.loadtxt('dota.csv',delimiter=',',skiprows=2,dtype='object') 
data=data[['embedding']]
#print(data)
#data[['embedding']]
#print(data.describe())
lst.append(data)
print(float(data[['embedding']]))