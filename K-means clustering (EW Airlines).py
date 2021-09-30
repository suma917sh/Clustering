#!/usr/bin/env python
# coding: utf-8

# K-means

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# In[3]:


airlines= pd.read_excel("C:\\Users\\hp\\Downloads\\EastWestAirlines.xlsx")


# In[5]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_airlines_df = scaler.fit_transform(airlines.iloc[:,1:])


# In[6]:


# How to find optimum number of  cluster
#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:


# In[7]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_airlines_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[8]:


#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(4, random_state=42)
clusters_new.fit(scaled_airlines_df)


# In[9]:


clusters_new.labels_


# In[10]:


#Assign clusters to the data set
airlines['clusterid_new'] = clusters_new.labels_


# In[11]:


#these are standardized values.
clusters_new.cluster_centers_


# In[12]:


airlines.groupby('clusterid_new').agg(['mean']).reset_index()


# In[13]:


airlines


# In[ ]:




