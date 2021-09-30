#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Import the libraries
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[34]:


df = pd.read_csv("C:\\Users\\hp\\Downloads\\crime_data1.csv");
print(df.head())


# In[35]:


print(df.info())


# In[36]:


df.drop(df.columns[0],axis=1,inplace=True)


# In[37]:


array=df.values
array


# In[38]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X


# In[39]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[30]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[32]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl


# In[33]:


pd.concat([df,cl],axis=1)


# In[ ]:




