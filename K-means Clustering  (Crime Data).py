#!/usr/bin/env python
# coding: utf-8

# K-Means

# In[14]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# In[15]:


crime= pd.read_csv("C:\\Users\\hp\\Downloads\\crime_data.csv")


# In[16]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_crime_df = scaler.fit_transform(crime.iloc[:,1:])


# In[17]:


# How to find optimum number of  cluster
#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:


# In[18]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_crime_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[19]:


#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(4, random_state=42)
clusters_new.fit(scaled_crime_df)


# In[20]:


clusters_new.labels_


# In[21]:


#Assign clusters to the data set
crime['clusterid_new'] = clusters_new.labels_


# In[22]:


#these are standardized values.
clusters_new.cluster_centers_


# In[23]:


crime.groupby('clusterid_new').agg(['mean']).reset_index()


# In[24]:


crime


# In[ ]:




