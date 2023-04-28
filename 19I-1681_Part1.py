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


# In[2]:


# Fixing random state for reproducibility
np.random.seed(19680801)

def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin


# In[3]:


n_samples = 200
random_state = 170
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

X, y = varied

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

df = pd.DataFrame(X, columns=['X1','X2'])
# df['y'] = y

df.to_csv("generatedDataset.csv")

x1 = df['X1']
x2 = df['X2']


# In[4]:


colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']), int(max(y) + 1))))
plt.scatter(x1, x2, s=20, color=colors[y])


# In[5]:


df


# In[6]:


def K(x, y, mu_x, mu_y, sig_x, sig_y, rho):
    z = (((x-mu_x)**2)/sig_x**2) - ((2*rho*(x-mu_x)*(y-mu_y))/sig_x*sig_y) + ((y-mu_y)/mu_y**2)
    
    epsilon = 1 - (rho**2)
    
    denum = 2 * math.pi * sig_x * sig_y * (math.sqrt(epsilon))
    t1 = 1 / denum
    t2 = -(z / (2 * epsilon))        

    res = t1 * (np.exp(t2))
    
    return res


# In[7]:


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


# In[8]:


den = F(df,2.5,2.5,0)
df['Density'] = den
df['y'] = y


# In[9]:


den


# In[10]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

xs = x1
ys = x2
zs = den
# ax.scatter(xs, ys, zs, marker='o')
# ax.scatter(xs, ys, zs, marker='^')

ax.scatter(xs, ys, zs)

# For each set of style and range settings, plot n random points in the box
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#     xs = x1
#     ys = x2
#     zs = den
#     ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Densities')

plt.show()


# In[ ]:




