
# coding: utf-8

# In[5]:


# import required packages

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pydotplus

from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


# In[2]:


os.chdir("C:/Users/shukl/AppData/Local/Programs/Python/Untitled Folder")
Crime_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv.', na_values = [None, 'NaN','Nothing'], header = 0) 
Crime_2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv.', na_values = [None, 'NaN','Nothing'], header = 0) 
Crime_3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv.', na_values = [None, 'NaN','Nothing'], header = 0)


# In[3]:


Crime_Data = [Crime_1, Crime_2, Crime_3]
Crime_Data = pd.concat(Crime_Data, axis = 0)


# In[6]:


# remove duplicates
Crime_Data.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
Crime_Data.drop(['Unnamed: 0','Case Number','IUCR','FBI Code','Updated On','Location',
                 'X Coordinate','Y Coordinate','Location'], inplace = True, axis = 1)


# In[7]:


Crime_Data.Date = pd.to_datetime(Crime_Data.Date, format = '%m/%d/%Y %I:%M:%S %p')
Crime_Data.index = pd.DatetimeIndex(Crime_Data.Date)


# In[8]:


Crime_Data['Primary Type'] = pd.Categorical(Crime_Data['Primary Type'])
Crime_Data['Description'] = pd.Categorical(Crime_Data['Description'])
Crime_Data['Location Description'] = pd.Categorical(Crime_Data['Location Description'])


# In[9]:


Crime_Data = Crime_Data.dropna(axis = 0, how = 'any')
Crime_Data = Crime_Data[Crime_Data.Longitude != '-87.1:00:00 AM']


# In[10]:


Arrest_Data = Crime_Data.drop('Arrest', axis = 1)
Arrest_Data = Arrest_Data.drop('Date', axis = 1)
Arrest_Data = Arrest_Data.drop('Block', axis = 1)
Arrest_Target = Crime_Data['Arrest']


# In[11]:


Arrest_Data['Primary Type'] = (Arrest_Data['Primary Type']).cat.codes
Arrest_Data['Location Description'] = (Arrest_Data['Location Description']).cat.codes
Arrest_Data['Description'] = (Arrest_Data['Description']).cat.codes

names = []
names = list(Arrest_Data)


# In[12]:


# Naive Bayes
gnb = GaussianNB()
scores_nb = cross_val_score(gnb, Arrest_Data, Arrest_Target)

print('Mean Cross Validation Accuracy for Naive Bayes: {}'.format(scores_nb.mean()))


# In[14]:


# Decision Tree
tree_entropy = DecisionTreeClassifier(criterion="entropy")
scores_dt = cross_val_score(tree_entropy, Arrest_Data, Arrest_Target)

print('Mean Cross Validation Accuracy for Decision Tree: {}'.format(scores_dt.mean()))


# In[ ]:


# Visualize data
model = tree_entropy.fit(Arrest_Data, Arrest_Target)
dot_data = export_graphviz(tree_entropy, feature_names = names, out_file=None, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)


# In[ ]:


# KNN
#normalize data
mms = MinMaxScaler()
Arrest_Data_norm = mms.fit_transform(Arrest_Data)

knn = KNeighborsClassifier(n_neighbors=3)
scores_knn = cross_val_score(knn, Arrest_Data_norm, Arrest_Target)

print('Mean Cross Validation Accuracy for KNN: {}'.format(scores_knn.mean()))


# In[ ]:


# SVC
#standardize data
ss = StandardScaler()
Arrest_Data_scaled = ss.fit_transform(Arrest_Data)

svc = LinearSVC()
scores_svc = cross_val_score(svc, Arrest_Data_scaled, Arrest_Target)

print('Mean Cross Validation Accuracy for SVC: {}'.format(scores_svc.mean()))


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
scores_logreg = cross_val_score(logreg, Arrest_Data, Arrest_Target)
scores_logreg.mean()

print('Mean Cross Validation Accuracy for Logistic Regression: {}'.format(scores_logreg.mean()))


# In[ ]:


# split the dataset into train(70%) and test(30%)
X_train, X_test, y_train, y_test = train_test_split(Arrest_Data, Arrest_Target, test_size = 0.3)

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

# visualizing and optimizing hyperparameters for knn classifier using train and test sample
training_accuracy = []
test_accuracy = []
neighbors= range(1,100)

for opt_k in neighbors:
    # Build the model
    knn_clf = KNeighborsClassifier(n_neighbors = opt_k)
    knn_clf.fit(X_train_norm, y_train)
    
    # Record training set accuracy
    trainAccuracy = knn_clf.score(X_train_norm, y_train)
    training_accuracy.append(trainAccuracy)
    
    # Record test set accuracy
    testAccuracy = knn_clf.score(X_test_norm, y_test)
    test_accuracy.append(testAccuracy)

# Visualize train and test accuracy
plt.plot(neighbors, training_accuracy, label = "Training Accuracy")
plt.plot(neighbors, test_accuracy, label = "Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Number of neighbors")
plt.legend()


# In[ ]:


X_trainval, X_test, y_trainval, y_test = train_test_split(Arrest_Data, Arrest_Target)

# tuning hyperparameters for svc using cross-validation
best_score_svc = 0
for opt_c in [0.001, 0.01, 0.1, 1, 10, 100]:
    svm = SVC(C = opt_c)
    fold_accuracies_svc = cross_val_score(svm, X_trainval, y_trainval)
    score_svc = fold_accuracies_svc.mean()
    if score_svc > best_score_svc:
        best_param_svc = {'C': opt_c}
        best_score_svc = score_svc
        
svm_opt = SVC(**best_param_svc)
svm_opt.fit(X_trainval, y_trainval)
test_score_svc = svm_opt.score(X_test, y_test)
print("Best Score on validation set: {:.2f}".format(best_score_svc))
print("Best parameters: {}".format(best_param_svc))
print("Test set score: {:.2f}".format(test_score_svc))


# In[ ]:


# tuning hyperparameters for decision tree using cross-validation
best_score_tree = 0
for i in range(1, 5):
    for j in range(1, 10):
        tree = DecisionTreeClassifier(criterion="entropy", max_depth = i, max_leaf_nodes = j)
        fold_accuracies_tree = cross_val_score(tree, X_trainval, y_trainval)
        score_tree = fold_accuracies_tree.mean()
        if score_tree > best_score_tree:
            best_param_tree = {'criterion' : "entropy", 'max_depth' : i, 'max_leaf_nodes' : j}
            best_score_tree = score_tree
        
tree_opt = SVC(**best_param_tree)
tree_opt.fit(X_trainval, y_trainval)
test_score_tree = tree_opt.score(X_test, y_test)
print("Best Score on validation set: {:.2f}".format(best_score_tree))
print("Best parameters: {}".format(best_param_tree))
print("Test set score: {:.2f}".format(test_score_tree))

