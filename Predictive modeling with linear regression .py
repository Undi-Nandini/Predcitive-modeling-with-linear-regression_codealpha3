#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"C:\Users\nandi\Downloads\house.csv")


# In[3]:


data


# In[4]:


data.head(5)


# In[5]:


data.columns


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


#visulazation
data.isnull().sum()


# In[9]:


sns.relplot(x="price",y="bedroom_count",data=data)


# In[10]:


sns.relplot(x="price",y="net_sqm",data=data)


# In[11]:


sns.relplot(x="price",y="center_distance",data=data)


# In[12]:


sns.relplot(x="price",y="metro_distance",data=data)


# In[13]:


sns.relplot(x="price",y="floor",data=data)


# In[14]:


sns.pairplot(data)


# In[15]:


corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


# In[16]:


sns.scatterplot(x='bedroom_count', y='price', data=data)


# In[17]:


sns.boxplot(x='floor', y='price', data=data)


# In[18]:


sns.histplot(data['price'], kde=True)


# In[19]:


#modelbuilding
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[20]:


X=data.drop(["price"],axis=1)
Y=data["price"]


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[22]:


linreg=LinearRegression()


# In[23]:


linreg.fit(X_train, y_train)


# In[24]:


predict=linreg.predict(X_test)
predict


# In[25]:


linreg.score(X_test,y_test)


# In[26]:


from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
mse = mean_squared_error(y_test, predict)
mae = mean_absolute_error(y_test, predict)
r2 = r2_score(y_test, predict)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')


# In[27]:


residuals = y_test - predict
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')


# In[28]:


coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': linreg.coef_})
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)
print(coefficients)


# In[29]:


from sklearn.model_selection import cross_val_score
cross_val_scores = cross_val_score(linreg, X, Y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation Mean Squared Error: {-cross_val_scores.mean()}')


# In[ ]:




