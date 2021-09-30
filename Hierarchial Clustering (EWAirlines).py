#!/usr/bin/env python
# coding: utf-8

# Hierarchial Clustering

# In[1]:


#import hierarchial clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


Airlines=pd.read_excel("C://Users//hp//Downloads//EastWestAirlines.xlsx")


# In[3]:


Airlines.head(25)


# In[4]:


#Normalization Function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# In[7]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airlines.iloc[:,1:])


# In[8]:


# create dendrogram
dendrogram=sch.dendrogram(sch.linkage(df_norm,method='complete'))


# In[9]:


#Create clusters
hc=AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')


# In[10]:


#Save clusters for chart
y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])

Clusters


# In[11]:


data=pd.concat([Airlines, Clusters],axis=1)


# In[12]:


data


# In[13]:


data.groupby('Clusters').mean()


# In[ ]:





# In[ ]:




