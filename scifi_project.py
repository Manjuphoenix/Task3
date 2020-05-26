#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[14]:


data =  load_boston()


# In[15]:


print(data.DESCR)


# In[16]:


dataset = pd.DataFrame(data.data, columns = data.feature_names)


# In[17]:


target = pd.DataFrame(data.target, columns=['target'])


# In[18]:


target


# In[19]:


dataset.corr


# In[20]:


X_train, X_test, y_train, y_test = train_test_split( dataset, target, test_size=0.15, random_state=42)


# In[22]:


model = LinearRegression()


# In[23]:


model.fit(X_train, y_train)


# In[26]:


y_pred = model.predict(X_test)



# In[30]:


error = mean_squared_error(y_test, y_pred)


print(error)



