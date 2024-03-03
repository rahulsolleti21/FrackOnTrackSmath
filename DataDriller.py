#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


# In[2]:


# This is the ready-to-use version of that data already converted to CSV.
url = 'https://raw.githubusercontent.com/yohanesnuwara/energy-analysis/main/data/SPE_ML_data.csv'

wells = pd.read_csv(url)
wells.head()


# In[3]:


# Summary stats of data
wells.describe()


# In[4]:


def sort_values(df, column_to_sort, ascending=True):
  sorted_df = df.sort_values(column_to_sort, ascending=ascending)
  sorted_df = sorted_df[['Lease', column_to_sort]]
  return sorted_df


# In[5]:


# Sort leases from largest to smallest cluster per stage
sorted_df = sort_values(wells, '# Clusters per Stage', ascending=False)

sorted_df.head(10)


# In[6]:


import pandas as pd

def corrcoef_heatmap(df, vmin=-0.5, vmax=0.5):

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    
    # Generate correlation matrix
    corr = numeric_cols.corr(method='spearman')

    # Generate a mask and draw heatmap
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(12,10))
    sns.heatmap(corr, mask=mask, cmap='Spectral_r', 
                vmin=vmin, vmax=vmax, square=True)
    plt.show()


# In[7]:


corrcoef_heatmap(wells, vmin=-0.8, vmax=0.8)


# In[19]:


def t_test(N, alpha=0.05):
  # Default 5% confidence level
  t = sp.stats.t.isf(alpha/2, N-2)
  r_crit = t/np.sqrt((N-2)+ np.power(t,2))
  return r_crit


# In[20]:


# t-test
N = len(wells) # number of leases
r_crit = t_test(N)

r_crit


# In[21]:


# Drop uncorrelated features
uncorrelated_features = ['WellboreDiameter(ft)', 'Porosity', 'WaterSaturation', 
                         'SepTemperature(degF)', 'SepPressure(psi)',
                         '#Stages', '#Clusters', 'SandfaceTemp(degF)',
                         'StaticWellheadTemp(degF)', 'CondensateGravity(API)',
                         'H2S', '#ofTotalProppant(Lbs)']

wells = wells.drop(uncorrelated_features, axis=1, errors='ignore')


# In[22]:


# Encode formation column
le = LabelEncoder()
wells['Formation/Reservoir'] = le.fit_transform(wells['Formation/Reservoir'].values)

wells.head()


# In[23]:


# Splitting features and targets
df = wells.iloc[:,1:] # Ignoring lease name

target_feature = '# Clusters per Stage'
X = df.drop([target_feature], axis=1)
y = df[target_feature]
     


# In[24]:


def scores(param_name, param_range):
  model = DecisionTreeRegressor(random_state=5)
  scorer = make_scorer(mean_squared_error)

  # LOOCV is CV with N folds. N number of data = 43
  train_scores, test_scores = validation_curve(model, X, y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               cv=53, scoring=scorer)

  train_score = np.mean(train_scores, axis=1)
  test_score = np.mean(test_scores, axis=1)  
  return train_score, test_score 


# In[25]:


plt.rcParams['font.size'] = 20

# Range for parameters
param_range = [1,2,3,4,5,6,7,8,9,10]

# Scores for varying max_depth
train_score1, test_score1 = scores('max_depth', param_range)

# Scores for varying min_samples_leaf
train_score2, test_score2 = scores('min_samples_leaf', param_range)

# Plot validation curves
plt.figure(figsize=(9,4))

plt.subplot(1,2,1)
plt.plot(param_range, train_score1, label='train')
plt.plot(param_range, test_score1, label='test')
plt.ylim(0,5.2)
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.legend(loc='lower right', fontsize=15)

plt.subplot(1,2,2)
plt.plot(param_range, train_score2, label='train')
plt.plot(param_range, test_score2, label='test')
plt.ylim(0,5.2)
plt.xlabel('min_samples_leaf')
plt.legend(loc='lower right', fontsize=15)

plt.tight_layout()
plt.show()


# In[26]:


# Instantiate decision tree model
dt = DecisionTreeRegressor(random_state=5)
# Fit the model
dt.fit(X, y)

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred[0]

# Get the test data
X_test = df.drop([target_feature], axis=1)

# Predict using the model
y_pred = predict(dt, X_test)





# In[27]:


column_names = X.columns.tolist()
print(column_names)


# In[28]:


def visualize(model):

    feature_names = X.columns.values.tolist()
    
    fig = plt.figure(figsize=(35,20))
    _= tree.plot_tree(model, feature_names=feature_names, filled=True, fontsize=13)

visualize(dt)


# In[ ]:




