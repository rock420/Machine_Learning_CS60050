
#Roll- 17EC10006    #Name- Ayan Saha    #Assignment No-3       #python3

import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

##########################################################  Decision Tree  ###################################################################

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
    def __init__(self,attribute):
        self.root=None
        self.attribute=attribute
        
    def build_tree(self,training,features,max_tree_depth=4,value=None,l=0):
        
        p1=list(training[:,3]).count('yes')
        p0=list(training[:,3]).count('no')
        
        if(p1==0): 
            self.root=node(training,value,leaf=True,predict='no')
            return self.root
        if(p0==0): 
                self.root=node(training,value,leaf=True,predict='yes')
                return self.root
        
        if(len(features)==0 ):
            if(p0>p1): 
                self.root=node(training,value,leaf=True,predict='no')
                return self.root
            else: 
                self.root=node(training,value,leaf=True,predict='yes')
                return self.root
        
        max_gain=-1
        splitting_attr=''
        for i in features.keys():
            ig=gain(training,self.attribute[i])
            if(ig>max_gain):
                max_gain=ig
                splitting_attr=i
        branches=[]
       
        for v in features[splitting_attr]:
            if(training[training[:,self.attribute[splitting_attr]]==v].size==0): 
                    if(p0>p1): branches.append(node(training,value=v,leaf=True,predict='no'))
                    else: branches.append( node(training,value=v,leaf=True,predict='yes'))
            else:
                f={key:val for key,val in features.items() if key!=splitting_attr}
                branches.append(self.build_tree(training[training[:,self.attribute[splitting_attr]]==v],f,
                                      max_tree_depth,value=v,l=l+1) )
            
        root=node(training,value,splitting_attr,branches)
        self.root=root
        return root
  
    def Prediction(self,test):
        r=self.root
        while(r.leaf!=True):
            a=r.splitting_attr
            for node in r.branches:
                if(test[self.attribute[a]]==node.value):
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

##############################################################################################################################################

data=pd.read_csv('data3_19.csv')
test=pd.read_csv('test3_19.csv',header=None)

training=data.to_numpy()
test_data=test.to_numpy()
N=training.shape[0]
indices=[i for i in range(0,N)]
init_prob=1/N
weight=[init_prob for i in range(0,N)]
T=5   # No of Classifier
features={}
for feature in data.columns.values[:3]:
    features[feature]=data[feature].value_counts().index.values
attribute={'pclass':0,'age':1,'gender':2,'survived':3}
target={'yes':1,'no':-1}
Trees=[]
Classifier_Weight=[]
                         ################# Adaboost ##################

for i in range(0,T):
    sample_indices=np.random.choice(indices,N,p=weight)
    sample_data=training[[sample_indices]]
    tree=Tree(attribute)
    root=tree.build_tree(sample_data,features)
    #tree.print_tree(root)
    error=0
    misclassified=[]
    correctclassified=[]
    for j in range(0,N):
        if(sample_data[j,3]!=tree.Prediction(sample_data[j])):
            error+=weight[sample_indices[j]]
            misclassified.append(sample_indices[j])
        else:
            correctclassified.append(sample_indices[j])
   
    if(error==0): error=1e-4       # to avoid log(infinity)
    #print(error)
    alpha=(1/2)*math.log(abs(1-error)/error)
    for j in misclassified:
        weight[j]=weight[j]*math.exp(alpha)
    for j in correctclassified:
        weight[j]=weight[j]*math.exp(-alpha)
    s=sum(weight)
    weight=[k/s for k in weight]
    
    Trees.append(tree)
    Classifier_Weight.append(alpha)

def Predict(test):
    Predictions=[]
    s=0
    for instance in test:
        Classifier=0
        for i in range(len(Trees)):
            Classifier+=Classifier_Weight[i]*target[Trees[i].Prediction(instance)]
        if Classifier>=0:
            Predictions.append('yes')
        else: Predictions.append('no')
        
        if(Predictions[-1]==instance[3]): s+=1
    print(s/len(test))
        
    return Predictions


print("Training Accuracy")
p_train=Predict(training)
print("Test Accuracy")
p_test=Predict(test_data)

