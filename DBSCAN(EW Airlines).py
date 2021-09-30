#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Import the libraries
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


df = pd.read_excel("C:\\Users\\hp\\Downloads\\EastWestAirlines.xlsx");

print(df.head())


# In[10]:


print(df.info())


# In[11]:


df.drop(['ID','Award'],axis=1,inplace=True)


# In[12]:


array=df.values
array


# In[13]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X


# In[14]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[15]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[16]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl


# In[17]:


pd.concat([df,cl],axis=1)


# In[ ]:




