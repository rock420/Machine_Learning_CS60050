
# Roll- 17EC10006     # Name- Ayan Saha     # Assignment No- 4       # python3 

import numpy as np
import pandas as pd

data=pd.read_csv("data4_19.csv",header=None)
data.columns=['sepal_length','sepal_width','petal_length','petal_width','cluster']



def Jaccard_dist(cluster_group,ground_truth):          ## function for Jaccard Distance between obtained clusters and Ground truth clusters
    for k in range(K):
        intersection=[v for v in cluster_group[k] if v in ground_truth[k]]
        union=[v for v in cluster_group[k] if v not in ground_truth[k]]
        union=union+ground_truth[k]
        
        if(len(cluster_group[k])==0 and len(ground_truth[k])==0): Jaccard_index=1
        else: Jaccard_index=len(intersection)/len(union)
        Jaccard_distance=1-Jaccard_index
        
        print("Jaccard Distance of {} cluster: {}".format(cluster_codetoname[k],Jaccard_distance))



############################################## Variables ##############################################
K=3
X=data.drop(['cluster'],axis=1).to_numpy()
m=X.shape[0]
cluster_nametocode={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
cluster_codetoname={0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
iterations=50
ground_truth={0:[],1:[],2:[]}
for i in range(m):
    ground_truth[cluster_nametocode[data.iloc[i,4]]].append(data.iloc[i,:-1].tolist())


############################################### k-means Algorithm ##############################################
idx=np.random.randint(m,size=3)
centroid=X[idx,:]
cluster=[-1 for i in range(m)]
for i in range(iterations):
      cluster_group={0:[],1:[],2:[]}
      for j in range(m):                           ### Cluster Assigning
          min_dist=np.inf
          for k in range(K):
              dist=np.linalg.norm(centroid[k]-X[j])
              if(dist<min_dist): 
                  min_dist=dist
                  cluster[j]=k
          cluster_group[cluster[j]].append(X[j].tolist())

      for k in range(K):                         ### Centroid Assigning
          centroid[k]=np.array(cluster_group[k]).mean(axis=0)

count=0
for i in range(m):
	if cluster_nametocode[data.iloc[i,4]]==cluster[i]:
		count+=1
	print(i+1," ",cluster_codetoname[cluster[i]]," ",data.iloc[i,4])
print(count)

#print("\n")      
for k in range(K):
	print("Centroid of {} cluster: {}".format(cluster_codetoname[k],centroid[k]))
print("\n")
Jaccard_dist(cluster_group,ground_truth)

############################################### end ###########################################################
