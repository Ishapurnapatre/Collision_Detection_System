#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                        ## For plotting graphs
import seaborn as sns                                  ## Data Visualization Library
import re                                              ## Used for string matching, searching, and manipulation based on patterns.
import random                                          ## Used for random sampling, shuffling, and generating random values.
from datetime import timedelta                         ## Useful for date and time arithmetic
from sklearn.preprocessing import LabelEncoder         ## Convert categorical labels into numeric labels
from sklearn.model_selection import train_test_split   ## For splitting the dataset into training and test sets
from matplotlib.ticker import FuncFormatter            ## A class for custom formatting of axis ticks.
from sklearn.preprocessing import StandardScaler       ## Library for standardizing the features (scaling)
from sklearn.metrics import silhouette_score           ## Used in clustering to evaluate the quality of clusters.
from sklearn.cluster import KMeans                     ## Used for partitioning the data into K clusters.
import plotly.express as px      


# In[2]:


data = pd.read_csv('sensor_data.csv')
df = pd.DataFrame(data)  
df 


# In[4]:


# Acceleration plot
ax = sns.lineplot(x='id', y='accel_x', data=df)
plt.show()
ax = sns.lineplot(x='id', y='accel_y', data=df)
plt.show()
ax = sns.lineplot(x='id', y='accel_z', data=df)
plt.show()


# In[5]:


# Gyroscope plot
ax = sns.lineplot(x='id', y='gyro_x', data=df)
plt.show()
ax = sns.lineplot(x='id', y='gyro_y', data=df)
plt.show()
ax = sns.lineplot(x='id', y='gyro_z', data=df)
plt.show()


# In[ ]:




