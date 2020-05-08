# Roll-17EC10006        # Name- Ayan Saha    #Assignment number 2      # python3

import pandas as pd
import numpy as np

def read(path):
    raw=pd.read_csv(path).to_numpy()

    for i in range(raw.shape[0]):
        L=list(map(int,raw[i][0].split(',')))
        if i==0:
            data=np.asarray([L])
        else:
            data=np.append(data,np.asarray([L]),axis=0)
    return data

class Naive_Bayes():
    
    def __init__(self):
        self.prob={}
        self.predictions=[]
        
    def separate(self,data):
        separated={}
        for i in range(len(data)):
            instance=data[i]
            if(instance[0] not in separated.keys()):
                separated[instance[0]]=np.asarray([instance])
            else: separated[instance[0]]=np.append(separated[instance[0]],[instance],axis=0)
        return separated

    def fit(self,data):
        separated=self.separate(data)
        
        n0=len(separated[0])
        n1=len(separated[1])
        self.prob["p0"]=n0/(n0+n1)
        self.prob["p1"]=n1/(n0+n1)
        
        
        for i in range(1,data.shape[1]):
            s1="p_X"+str(i)+"_D0"
            s2="p_X"+str(i)+"_D1"

            unique,counts=np.unique(separated[0][:,i],return_counts=True)
            counts=(counts+1)/(n0+5)   #Laplacian Smoothing
            d=dict(zip(unique,counts))  
            for j in range(1,6):   # as value range of each attribute is [1,5]
                if j not in d.keys():
                    d[j]=1/(n0+5)   #count=0
            self.prob[s1]=d 

            unique,counts=np.unique(separated[1][:,i],return_counts=True)
            counts=(counts+1)/(n1+5)
            d=dict(zip(unique,counts))
            for j in range(1,6):
                if j not in d.keys():
                    d[j]=1/(n1+5)
            self.prob[s2]=d
        
    def predict(self,test):
        count=0
        for i in range(test.shape[0]):
            instance=test[i]
            p_0=1
            p_1=1
            for j in range(1,test.shape[1]):
                s1="p_X"+str(j)+"_D0"
                s2="p_X"+str(j)+"_D1"
                p_0=p_0*self.prob[s1][instance[j]]
                p_1=p_1*self.prob[s2][instance[j]]
            p_0=p_0*self.prob["p0"]
            p_1=p_1*self.prob["p1"]
            if(p_0>p_1):
                pred=0
            else:
                pred=1
            self.predictions.append(pred)
            if(instance[0]==pred): count+=1
            
        accuracy=count/(test.shape[0])
        print("count: {}/{}".format(count,test.shape[0]))
        print("accuracy: {0:.2f}%".format(accuracy*100))


train=read("data2_19.csv")
test=read("test2_19.csv")
model=Naive_Bayes()
model.fit(train)
model.predict(test)
