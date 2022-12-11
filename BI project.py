#!/usr/bin/env python
# coding: utf-8

# # Apriori

# ## Importing the libraries

# In[1]:


get_ipython().system('pip install apyori')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


#Reading our data
inp = input("Write your data path, please?\n")
dataset = pd.read_csv(inp)
dataset.head()


# ## Data Preprocessing

# In[4]:


# First, We will make our list of lists
asso = []
num_of_rows = dataset.shape[0]
num_of_columns= dataset.shape[1]
for i in range(0, num_of_rows):
    asso.append([str(dataset.values[i,j]) for j in range(0, num_of_columns)])


# In[5]:


asso


# ## Training the Apriori model on the dataset

# In[6]:


from apyori import apriori
rules = apriori(transactions = asso, min_lift = 2, min_lenght = 2, max_length = 3)


# ## Visualising the results

# ### Displaying the first results coming directly from the output of the apriori function

# In[7]:


results = list(rules)
results


# ### Putting the results well organised into a Pandas DataFrame

# In[8]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side',
                                                               'Support', 'Confidence', 'Lift'])


# ### Displaying the results non sorted

# In[9]:


resultsinDataFrame


# >**According to our model, females are most likely to be survived (primarily adult females)**

# ## Additional part

# In[10]:


# Let's prove these information


# In[11]:


dataset.head()


# In[12]:


dataset.groupby('Sex').agg({'Survived': "value_counts"})


# In[13]:


print("Number of adult females survived:",(dataset[(dataset['Sex'] == "Female") & (dataset['Survived'] == "Yes")].count()[0]))
print("Number of adult females didn't survive:",dataset[(dataset['Sex'] == "Female") & (dataset['Survived'] == "No")].count()[0])

