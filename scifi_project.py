#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import smtplib, ssl


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
error = 100-error

print(error)


port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "manjunath1055@gmail.com"  # Enter your address
receiver_email = "manjunath.d1@visioneer.atria.edu "  # Enter receiver address
password = "*********"
message = """Subject: Hi there
This is jenkins reporting,
your model accuracy is:"""+ str(error) +"""%"""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)


