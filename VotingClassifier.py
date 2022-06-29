#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#導入資料
df = pd.read_csv('BloodData2.csv')


# In[2]:


#載入模型
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ## 建立模型

# In[3]:


from sklearn.ensemble import VotingClassifier

#model1 = SVC('linear',probability=True)
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)

model3 = KNeighborsClassifier(n_neighbors=1)

model = VotingClassifier(estimators=[ ('lr',model1),('dt', model2),('knn', model3)], voting='hard')


# In[4]:


#先確認資料INDEX 
print(df.columns)
df.head()


# ## 處理資料
# 

# In[5]:


#載入標準化比例尺（StandardScaler）套件
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('Target',axis=1))
scaled_features = scaler.transform(df.drop('Target',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[6]:


from sklearn.model_selection import train_test_split

X = df_feat
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)


# ## 訓練模型

# In[7]:


model.fit(X_train,y_train)


# In[9]:


model.score(X_test,y_test)


# In[92]:


#測試好壞
from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# ## KNN

# In[93]:


#K值等於9
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=9')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# ## 決策樹

# In[94]:


DT = tree.DecisionTreeClassifier(random_state=1)
DT.fit(X_train,y_train)

pred = DT.predict(X_test)

print('WITH DT')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# ## 邏輯回歸

# In[95]:


LR = LogisticRegression(random_state=1)
LR.fit(X_train,y_train)

pred = LR.predict(X_test)

print('WITH LR')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




