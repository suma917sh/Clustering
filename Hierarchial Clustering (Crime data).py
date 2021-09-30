#!/usr/bin/env python
# coding: utf-8

# Hierarchial Clustering
# 

# In[1]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


Crime=pd.read_csv("C://Users//hp//Downloads//crime_data.csv")


# In[3]:


Crime.head(25)


# In[4]:


#Normalization Function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# In[5]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Crime.iloc[:,1:])


# In[6]:


# create dendrogram
dendrogram=sch.dendrogram(sch.linkage(df_norm,method='complete'))


# In[7]:


#Create clusters
hc=AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')


# In[12]:


#Save clusters for chart
y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])

Clusters


# In[13]:


data=pd.concat([Crime, Clusters],axis=1)


# In[14]:


data


# In[15]:


data.groupby('Clusters').mean()


# In[ ]:





# In[24]:





# In[ ]:




