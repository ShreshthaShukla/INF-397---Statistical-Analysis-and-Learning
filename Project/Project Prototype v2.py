
# coding: utf-8

# In[1]:


# import required packages

import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


## LOAD DATA


# In[3]:


# display parent directory and working directory

print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
print(os.getcwd()+':', os.listdir(os.getcwd()));


# In[4]:


# import data

df1 = pd.read_csv("Chicago_Crimes_2001_to_2004.csv", index_col = 0)
df2 = pd.read_csv("Chicago_Crimes_2005_to_2007.csv", index_col = 0)
df3 = pd.read_csv("Chicago_Crimes_2008_to_2011.csv", index_col = 0)
df4 = pd.read_csv("Chicago_Crimes_2012_to_2017.csv", index_col = 0)


# In[5]:


# combine data frames

frames = [df1, df2, df3, df4]
full_dataset = pd.concat(frames)


# In[6]:


full_dataset.info()


# In[7]:


full_dataset.Year.unique()


# In[8]:


# fix similar variables

full_dataset['Year'].replace('2001', 2001,inplace=True)
full_dataset['Year'].replace('2002', 2002,inplace=True)
full_dataset['Year'].replace('2003', 2003,inplace=True)
full_dataset['Year'].replace('2004', 2004,inplace=True)


# In[9]:


full_dataset.Year.unique()


# In[10]:


# remove strange values

full_dataset2 = full_dataset[full_dataset.Year != 'Y Coordinate']


# In[11]:


full_dataset3 = full_dataset2[full_dataset2.Year != 41.789832136]


# In[ ]:


## DATA EXPLORATION


# In[ ]:


full_dataset3.sort_values(by=['Year'])
full_dataset3.Year.value_counts()


# In[ ]:


full_dataset3.rename(columns={'Primary Type': 'Primary_Type'}, inplace=True)
full_dataset3.head()


# In[ ]:


full_dataset3['Primary_Type'].replace('NON - CRIMINAL','NON-CRIMINAL',inplace=True)
full_dataset3['Primary_Type'].replace('NON-CRIMINAL (SUBJECT SPECIFIED)','NON-CRIMINAL',inplace=True)


# In[ ]:


full_dataset3.sort_values(by=['Primary_Type'])
full_dataset3.Primary_Type.value_counts()


# In[ ]:


full_dataset3.Arrest.unique()


# In[ ]:


full_dataset3['Arrest'].replace('True', True,inplace=True)
full_dataset3['Arrest'].replace('False', False,inplace=True)


# In[ ]:


full_dataset3.groupby('Year').Arrest.value_counts()


# In[ ]:


# finalizing dataset

final_dataset = full_dataset3[full_dataset3.Year != 2017]
final_dataset.info()


# In[ ]:


final_dataset.Year.value_counts()


# In[ ]:


## BUILDING MODELS


# In[ ]:


refined_data_set = final_dataset[['Date', 'IUCR', 'Primary_Type', 'Description', 'Location Description', 'Arrest', 'Domestic', 'Block']].copy()


# In[ ]:


refined_data_set.head()


# In[ ]:


# Prepare Train Test Split

from sklearn.model_selection import train_test_split
X = refined_data_set[['IUCR', 'Primary_Type', 'Description', 'Location Description', 'Domestic']].copy()
y = refined_data_set[['Arrest']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# In[ ]:


# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree_entropy = DecisionTreeClassifier(criterion="entropy").fit(X_train,y_train)

dot_data = tree.export_graphviz(tree_entropy, out_file = None, feature_names = list(X), class_names = list(y), filled = True, rounded = True)
graph = graphviz.Source(dot_data)
graph

