#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:18:28 2019

@author: ayan
"""

# Roll- 17EC10006      # Name- Ayan Saha    # Assignment no-1          # python3 with numpy and pandas 

import numpy as np
import pandas as pd   #only to read the data
import math


def entropy(S):
    p1=list(S[:,3]).count('yes')/(S.shape[0])
    p0=list(S[:,3]).count('no')/(S.shape[0])
    if(p0==0): return -p1*math.log2(p1)
    if(p1==0): return -p0*math.log2(p0)
    return -p0*math.log2(p0)-p1*math.log2(p1)

def gain(S,A):
    """
       S-> data
       A-> splitting attribute
    """
    unique_values=np.unique(S[:,A])
    S_len=S.shape[0]
    entropy_Sv=0
    for v in unique_values:
        Sv=S[S[:,A]==v]
        entropy_Sv+=(Sv.shape[0]/S_len)*entropy(Sv)
    
    return entropy(S)-entropy_Sv

class node:
    
    def __init__(self,data,value=None,splitting_attr=None,branches=[],leaf=False,predict=None):
        self.value=value
        self.splitting_attr=splitting_attr
        self.branches=branches
        self.leaf=leaf
        if(leaf):
            self.predict=predict
class Tree:
    def __init__(self,root=None):
        self.root=root
        
    def build_tree(self,training,features,max_tree_depth=4,value=None,l=0):
    
        p1=list(training[:,3]).count('yes')
        p0=list(training[:,3]).count('no')

        if(p1==0): return node(training,value,leaf=True,predict='no')
        if(p0==0): return node(training,value,leaf=True,predict='yes')
        if(features.shape[0]==0 ):
            if(p0>p1): return node(training,value,leaf=True,predict='no')
            else: return node(training,value,leaf=True,predict='yes')
       
        max_gain=-1
        splitting_attr=''
        for i in features:
            ig=gain(training,attribute[i])
            if(ig>max_gain):
                max_gain=ig
                splitting_attr=i
        branches=[]
        for v in np.unique(training[:,attribute[splitting_attr]]):
            if(training[training[:,attribute[splitting_attr]]==v].size==0): 
                    if(counts[0]>counts[1]): branches.append(node(training,value=v,leaf=True,predict='no'))
                    else: branches.append( node(training,value=v,leaf=True,predict='yes'))
            else:
                 branches.append(self.build_tree(training[training[:,attribute[splitting_attr]]==v],features[features!=splitting_attr],
                                      max_tree_depth,value=v,l=l+1) )
            
        root=node(training,value,splitting_attr,branches)
        self.root=root
        return root
  
    def Prediction(self,test):
        r=self.root
        while(r.leaf!=True):
            a=r.splitting_attr
            for node in r.branches:
                if(test[attribute[a]]==node.value):
                    r=node
                    break
        return r.predict
    def print_tree(self,root,ind='   ',l=0,splitting_attr=None):
        if(root.leaf==True):
            if(splitting_attr!=None):
                print(l*ind,splitting_attr," = ",root.value,": ",root.predict)
                return
            else: 
                print(l*ind,"survived = ",root.predict)
                return
        if(splitting_attr!=None): print(l*ind,splitting_attr," = ",root.value)
        
        for node in root.branches:
            self.print_tree(node,l=l+1,splitting_attr=root.splitting_attr)

data=pd.read_csv("data1_19.csv")
X=data.to_numpy()
indices = np.random.permutation(X.shape[0])
test_idx, training_idx = indices[:300], indices[300:]
training, test = X[training_idx,:], X[test_idx,:]
features=np.array(['pclass','age','gender'])
attribute={'pclass':0,'age':1,'gender':2,'survived':3}
        
tree=Tree()
root=tree.build_tree(training,features)
tree.print_tree(root)

s=0
for i in range(len(training)):
	if(training[i,3]==tree.Prediction(training[i])):
		s+=1
print("Training accuracy: {}".format(s/len(training)))

s=0
for i in range(len(test)):
	if(test[i,3]==tree.Prediction(test[i])):
		s+=1
print("correct test prediction: {}/300".format(s))
print("Test accuracy: {}".format(s/len(test)))
