# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:33:57 2019

@author: Srivatsan
"""
import csv
import copy
from collections import defaultdict
import numpy as np

columns = defaultdict(list) # each value in each column is appended to a list

with open('pl.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

lst=[columns[k]]

    

lst=np.array(lst, dtype=np.float32)
#print(lst)

tst=[]
#tst=lst.copy()
#tst=lst[0][:127]
l=len(lst[0])
n=l/127
n=int(n)
print(n)
i=0
while i<n:
    tst=lst[0][(i*127+i):((i+1)*127+i)]
    i=i+1
print(tst)
#while i<(len(lst)):
 #   tst[n]= lst[0][i]
  #  n=n+1
   # i=i+1
    #if(n==127):
     #   n=0
    

#print(tst[1]) 