#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


# ## Generating Dataset

# In[2]:


n_samples = 2000
random_state = 170
# varied = datasets.make_blobs(n_samples=n_samples,
#                              cluster_std=[1.0, 2.5, 0.5],
#                              random_state=random_state)

varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 1.5, 0.5],
                             random_state=random_state)

X, y = varied

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

df = pd.DataFrame(X, columns=['X1','X2'])
# df['y'] = y

df.to_csv("generatedDataset.csv")

x1 = df['X1']
x2 = df['X2']


# ## Plotting

# In[3]:


colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']), int(max(y) + 1))))
plt.scatter(x1, x2, s=20, color=colors[y])


# In[4]:


df


# In[5]:


def K(x, y, mu_x, mu_y, sig_x, sig_y, rho):
    z = (((x-mu_x)/sig_x)**2) - (((2*rho)*(x-mu_x)*(y-mu_y))/(sig_x*sig_y)) + (((y-mu_y)/mu_y)**2)
    
    epsilon = 1 - (rho**2)
    
    denum = 2 * math.pi * sig_x * sig_y * (math.sqrt(epsilon))
    t1 = 1 / denum
    t2 = -(z / (2 * epsilon))        

    res = t1 * (np.exp(t2))
    
    return res


# In[6]:


def F(df, sig_x, sig_y, rho):
    den = []
    for i in range(n_samples):
        ans = []
        for j in range(n_samples):            
            x,y = df['X1'][i] , df['X2'][i]
            mu_x , mu_y = df['X1'][j] , df['X2'][j]       
            ans.append(K(x, y, mu_x, mu_y, sig_x, sig_y, rho)) 
            
        s = np.mean(ans)
        
        den.append(s)
    return den


# In[7]:


den = F(df,0.2,0.2,0)
df['Density'] = den
df['y'] = y


# In[15]:


df


# In[9]:


# den


# # Plotting 3D Graph

# In[172]:


from collections import OrderedDict
import numpy as np
from scipy.spatial import Delaunay

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

x = x1
y = x2
z = den

ax.scatter(x, y, z, s=50, c=np.arange(n_samples), marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Densities')

plt.show()


# In[112]:


from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

def P(df,x,y):
    if (x < min(df['X1'])) or (x > max(df['X1'])) or y < min(df['X2']) or y > max(df['X2']):            
        return 0
#     interp = LinearNDInterpolator((df['X1'], df['X2']), df['Density'])
    interp = NearestNDInterpolator((df['X1'], df['X2']), df['Density'])    
    p = (interp(x,y))
    return p


# In[158]:


def coord(df,x,y):        
    alp = 2
    h = 0.1
    rnd = 1
    
    while(1):
        old_x = np.round(x,rnd)
        old_y = np.round(y,rnd)

#         delta = x
        
        der_x = (P(df,x+h,y) - P(df,x,y))/h 
        der_y = (P(df,x,y+h) - P(df,x,y))/h  
        
        x = x + (alp * der_x)  # new x
        y = y + (alp * der_y)  # new y    
        
        new_x = np.round(x,rnd)
        new_y = np.round(y,rnd)            
        
#         delta = x - delta    
        
        if old_x == new_x and old_y == new_y:            
            return new_x, new_y                        

    return new_x, new_y  


# In[159]:


l1 = []
l2 = []
for i in range(len(df)-0):
    x = df['X1'][i]
    y = df['X2'][i]
    X,Y = coord(df,x,y)   
    l1.append(X)
    l2.append(Y)
#     print('\n')
l = list(zip(l1,l2))    


# ## Finding Close Values (Determining Clusters)

# In[160]:


from collections import Counter
Counter(l)


# In[182]:


# df1 = pd.DataFrame(l1,l2,columns=[['c1','c2']])
# math.isclose(l1[],l2[])


# In[ ]:





# ## Plotting Final 3D Graph(with labels)

# In[ ]:


# points2D = np.vstack([u,v]).T
# tri = Delaunay(points2D)
# simplices = tri.simplices

# fig = ff.create_trisurf(x=x, y=y, z=z,
#                          simplices=simplices,
#                          title="Torus", aspectratio=dict(x=1, y=1, z=0.3))
# fig.show()


# In[163]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

x = l1
y = l2
z = den

ax.scatter(x, y, z, s=50, c=np.arange(n_samples), marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Densities')

plt.show()


# In[ ]:




