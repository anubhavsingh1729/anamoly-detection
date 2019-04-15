
# coding: utf-8

# In[76]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# In[77]:

dt1 = pd.read_csv("/home/ashappy208/Downloads/sessions/1be94a28-3bb4-4fd1-86b9-8919a90b7d12.csv")
dt2 = pd.read_csv("/home/ashappy208/Downloads/sessions/2d4177fa-aabf-4902-b43a-c3cad0e88607.csv")
dt3 = pd.read_csv("/home/ashappy208/Downloads/sessions/7c638235-b6ad-4b6c-a421-83c094034b43.csv")
dt4 = pd.read_csv("/home/ashappy208/Downloads/sessions/72a1a430-d1b5-4e07-8ab3-1dca8b645353.csv")
dt5 = pd.read_csv("/home/ashappy208/Downloads/sessions/99a3c10d-a606-4a48-a243-dcfb8a0b939d.csv")
dt6 = pd.read_csv("/home/ashappy208/Downloads/sessions/37075094-5beb-47bf-acbb-7045f878c19d.csv")
dt7 = pd.read_csv("/home/ashappy208/Downloads/sessions/a44f7e8c-ab3a-4087-9329-f1204d17ecd8.csv")
dt8 = pd.read_csv("/home/ashappy208/Downloads/sessions/b1fccc91-05e3-47aa-8f82-65b0af2d3a50.csv")
dt9 = pd.read_csv("/home/ashappy208/Downloads/sessions/d7d6daf8-cf6a-4b6b-a104-632804761146.csv")
dt10 = pd.read_csv("/home/ashappy208/Downloads/sessions/d67e606a-802d-4665-8bba-9d2a1eb03b9d.csv")
dt11 = pd.read_csv("/home/ashappy208/Downloads/sessions/fe0dd989-aaf6-409d-a13a-f36d199c7019.csv")

# In[78]:

dt1.describe()

# In[79]:

dt1['type'].unique()

# In[80]:


plt.scatter(dt1.time,dt1.scrollY,marker = 'x')


# In[81]:


plt.scatter(dt1.type,dt1.scrollY,marker = 'x')


# In[82]:


import seaborn as sns
from sklearn.cluster import KMeans


# In[83]:


#dt3.boxplot(column =['scrollY'], grid = False)
dt1.boxplot(column =['time'], grid = False)
sns.boxplot(dt1.scrollY,dt1.type)


# In[84]:


#new dataframe with only 'time' and 'scrollY' column
new1 = dt1.drop(['height','width','scrollX','type'],axis = 1)


# In[85]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(new1)
y_kmeans = kmeans.predict(new1)


# In[86]:


#q1 = dt3.scrollY.quantile(.25)
#q3 = dt3.scrollY.quantile(.70)


# In[87]:


plt.scatter(new1['time'],new1['scrollY'], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[88]:


'''from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(dt2.type)
dt2.type = y'''
new2 = dt2.drop(['height','width','scrollX','type'],axis = 1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(new2)
y_kmeans = kmeans.predict(new2)

plt.scatter(new2['time'],new2['scrollY'], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[89]:


'''from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(min_samples = 2, eps = 3)
clusters = outlier_detection.fit_predict(new2)
list(clusters).count(-1)'''


# In[90]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
cluster.fit_predict(new1)


# In[91]:


plt.scatter(new1['time'],new1['scrollY'], c=cluster.labels_, cmap='rainbow')


# In[92]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(new2)
y_kmeans = kmeans.predict(new2)

plt.scatter(new2['time'],new2['scrollY'], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[93]:


#Isolation forest to predict outliers
from sklearn.ensemble import IsolationForest


# In[94]:


clf = IsolationForest(n_estimators = 100,max_samples=256)
clf.fit(new1)


# In[64]:


clt = clf.predict(new1)
print("anomalies : ", list(clt).count(-1))


# In[38]:


plt.scatter(dt2.time,dt2.scrollY)


# In[49]:


new2 = dt2.drop(['height','width','scrollX','type'],axis = 1) 


# In[129]:


clf = IsolationForest(n_estimators = 100,max_samples=256)
clf.fit(new2)
clt = clf.predict(new2)
print("anomalies : ", list(clt).count(-1))

# In[132]:

scores = clf.decision_function(new2)
plt.figure(figsize=(12, 8))
plt.hist(scores, bins=50)


# In[51]:


plt.scatter(dt3.time,dt3.scrollY)


# In[53]:


new3 = dt3.drop(['height','width','scrollX','type'],axis = 1) 
kmeans = KMeans(n_clusters=2)
kmeans.fit(new3)
y_kmeans = kmeans.predict(new3)

plt.scatter(new3['time'],new3['scrollY'], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[123]:


clf = IsolationForest(n_estimators = 100,max_samples=256)
clf.fit(new3)
clt = clf.predict(new3)
print("anomalies : ", list(clt).count(-1))
scores = clf.decision_function(new3)


# In[67]:


plt.scatter(dt4.time,dt4.scrollY)


# In[69]:


new4 = dt4.drop(['height','width','scrollX','type'],axis = 1) 


# In[124]:


clf = IsolationForest(n_estimators = 100,max_samples=256)
clf.fit(new4)
clt = clf.predict(new4)
print("anomalies : ", list(clt).count(-1))


# In[71]:


plt.scatter(dt5.time,dt5.scrollY)


# In[133]:


new5 = dt5.drop(['height','width','scrollX','type'],axis = 1) 
#new5 = new5.iloc[:, :53]


# In[125]:


clf = IsolationForest(n_estimators = 100,max_samples=53)
clf.fit(new5)
clt = clf.predict(new5)
print("anomalies : ", list(clt).count(-1))


# In[127]:


scores = clf.decision_function(new5)
plt.figure(figsize=(12, 8))
plt.hist(scores, bins=50)


# In[139]:


plt.scatter(dt11.time,dt11.scrollY)


# In[140]:


new11 = dt11.drop(['height','width','scrollX','type'],axis = 1) 
clf.fit(new11)
clt = clf.predict(new11)
print("anomalies : ", list(clt).count(-1))

