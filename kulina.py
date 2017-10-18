
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn import preprocessing

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


# Input data
kitchen = pd.read_csv("kitchen.csv", error_bad_lines=False, engine="python", encoding = "ISO-8859-1")
customer = pd.read_csv("customer.csv", error_bad_lines=False, engine="python", encoding = "ISO-8859-1")

customer1 = pd.read_csv("customer.csv", error_bad_lines=False, engine="python", encoding = "ISO-8859-1")
kitchen1 = pd.read_csv("kitchen.csv", error_bad_lines=False, engine="python", encoding = "ISO-8859-1")


# In[3]:


# Preprocessing for grouping
kitchen.drop(['minCapacity'], axis=1, inplace=True)
kitchen.drop(['maxCapacity'], axis=1, inplace=True)
kitchen.drop(['tolerance'], axis=1, inplace=True)

customer.drop(['customersName'], axis=1, inplace=True)
customer.drop(['qtyOrdered'], axis=1, inplace=True)


lef = preprocessing.LabelEncoder()

#Create a new column with transformed values.
kitchen['kitchenName'] = lef.fit_transform(kitchen['kitchenName'])
print(kitchen)



# # Overview
# Kunci utama dari Efisiensi pengiriman adalah customer harus terassign ke kitchen yang terdekat dulu
# Kami melakukannya dengan menSort customer dari jarak yang paling jauh dari titik pusat customer (sum/total lat long)
# Solusi tersebut belum optimal,tapi mendekati
# Solusi optimal = Sort dari Outermost customer
# Customer kemudian di assign ke kitchen terdekatnya, apabila sudah full maka diassign ke kitchen kedua terdekat, dst
# Sehingga bisa didapat group berupa customer yang terassign ke suatu kitchen
# 
# Driver kemudian di assign per group berdasarkan degree dan jarak
# Di assign tidak hanya berdasarkan jarak untuk mengoptimalkan waktu pengiriman selama 1 jam

# # Grouping customer to the best kitchen

# In[4]:


# Find center point of customer, buat nyari
# long
long_centroid = sum(customer['long'])/len(customer)
# lat
lat_centroid = sum(customer['lat'])/len(customer)


# In[ ]:


# Find distance from customer point to central customer point
customer['distSort'] = np.sqrt( (customer.long-long_centroid)**2 + (customer.lat-lat_centroid)**2)

# Sort by longest distance
customer = data.sort_values(['distSort'], ascending=False)


# In[ ]:


# Data already sorted from outermost customer 
# For each row in the column,assign customer to the the nearest kitchen, 
# if the kitchen already full, assign customer to the second nearest kitchen and so on.

# BELUM SELESAI YANG INI
clusters = []

for row in customer['distSort']:
    #clusters.append(cluster)

    
data['cluster'] = clusters


# In[5]:


# Data visulization customer assigned to its kitchen

def visualize(data):
    x = data['long']
    y = data['lat']
    Cluster = data['cluster'] 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x,y,c=Cluster, cmap=plt.cm.Paired, s=10, label='customer')
    ax.scatter(kitchen['long'],kitchen['lat'], s=10, c='r', marker="x", label='second')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    plt.colorbar(scatter)

    fig.show()


# In[6]:


# Visualization Example customer assigned to kitchen (without following constraint)
# THIS IS ONLY EXAMPLE
y = kitchen['kitchenName']
X = pd.DataFrame(kitchen.drop('kitchenName', axis=1))

clf = NearestCentroid()
clf.fit(X, y)
pred = clf.predict(customer)

customer1['cluster'] = pd.Series(pred, index=customer1.index)
customer['cluster'] = pd.Series(pred, index=customer.index)


# In[7]:


visualize(customer)


# In[ ]:


# Count customer order assigned to Kitchen
dapurMiji = (customer1.where(customer1['cluster'] == 0))['qtyOrdered'].sum()
dapurNusantara = (customer1.where(customer1['cluster'] == 1))['qtyOrdered'].sum()
familiaCatering = (customer1.where(customer1['cluster'] == 2))['qtyOrdered'].sum()
pondokRawon = (customer1.where(customer1['cluster'] == 3))['qtyOrdered'].sum()
roseCatering = (customer1.where(customer1['cluster'] == 4))['qtyOrdered'].sum()
tigaKitchenCatering = (customer1.where(customer1['cluster'] == 5))['qtyOrdered'].sum()
ummuUwais = (customer1.where(customer1['cluster'] == 6))['qtyOrdered'].sum()

d = {'Dapur Miji': dapurMiji , 'Dapur Nusantara': dapurNusantara, 'Familia Catering': familiaCatering, 'Pondok Rawon': pondokRawon,'Rose Catering': roseCatering, 'Tiga Kitchen Catering': tigaKitchenCatering, 'Ummu Uwais': ummuUwais}


# In[ ]:


# print(customer.cluster.value_counts())


# In[ ]:


# Print sum of assigned
print(d) 


# # Assign driver in group based on degree and distance

# In[ ]:


# Get degree for each customer in the cluster

def getDegree(data):
    # distance
    # center long lat (start of routing)
    center_latitude = #Tiap Kitchen
    center_longitude = #Tiap Kitchen
    degrees = []
    degree = 0
    # For each row in the column,
    for row in data['longitude']:
        degrees = np.rint(np.rad2deg(np.arctan2((data['latitude']-center_latitude),(data['longitude']-center_longitude))))
       #center di pulogadung
    data['degrees'] = degrees
    return data


# In[ ]:


# Assign driver dari kitchen ke customer berdasarkan degree dan jarak
# Priority utama berdasarkan degree jadi gaada driver yang deket doang
# Tapi belum dipikir gimana bisa optimize waktu harus satu jam max, tapi seenggaknya driver udah agak rata jaraknya
# Kasus khusus apabila yg degree nya kecil jaraknya jauh banget, dia driver baru.

# BELUM SELESAI YANG INI

