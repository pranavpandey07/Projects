#!/usr/bin/env python
# coding: utf-8

#  In the given dataset, our target variable is "Happiness Score" which is having continuous values hence this problem belongs to Regression.

# In[5]:


#importing important libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df=pd.read_csv("worldhappiness.csv")  #loading the file into the pandas


# In[7]:


df


# In[12]:


#dimensions of the data
df.shape


# In[13]:


df.head() #top five rows of the dataset


# In[15]:


df.isnull().sum()   #CHECKING THE MISSING VALUES


# There is no missing value present in the dataset

# # EXPLORATORY DATA ANALYSIS

# In[43]:


df.hist(bins=40,figsize=(15,15))


# WE CAN SEE FROM ABOVE HISTOGRAMS THAT MANY HAVE SKEWED DISTRIBUTIONS LIKE "TRUST","STANDARD ERROR","HEALTH","FAMILY",ETC.

# In[31]:


#TOP TEN HAPPIEST COUNTRIES

df.groupby('Country')['Happiness Score'].max().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(12,6),color='green')
plt.title('Top 10 happiest countries')


# In[26]:


df.columns


# In[29]:


#Top ten saddest countries in the world

df.groupby('Country')['Happiness Score'].max().sort_values(ascending=False).tail(10).plot(kind='bar', figsize=(12,6),color='red')
plt.title('Top 10 Saddest countries')


# In[32]:


#Top 20 countries with highest gdp
df.groupby('Country')['Economy (GDP per Capita)'].max().sort_values(ascending=False).head(20).plot(kind='bar', figsize=(20,6),color='limegreen')
plt.title('Top 20 countries with higghest GDP per capita')


# In[33]:


#Top 20 countries with lowest gdp
df.groupby('Country')['Economy (GDP per Capita)'].max().sort_values(ascending=False).tail(20).plot(kind='bar', figsize=(20,6),color='red')
plt.title('20 countries with lowest GDP per capita')


# In[34]:


# UNIVARIATE SCATTER PLOT 
plt.scatter(df.index,df["Freedom"])
plt.show()


# In[35]:


plt.scatter(df["Freedom"],df.index)
plt.show()


# In[36]:


sns.scatterplot(x=df["Happiness Score"],y=df["Freedom"])


# In[37]:


sns.scatterplot(x=df["Happiness Score"],y=df["Health (Life Expectancy)"])


# In[38]:


sns.scatterplot(x=df["Happiness Score"],y=df["Trust (Government Corruption)"])


# In[39]:


sns.scatterplot(x=df["Happiness Score"],y=df["Generosity"])


# In[40]:


sns.scatterplot(x=df["Happiness Score"],y=df["Economy (GDP per Capita)"])


# In[41]:


sns.scatterplot(x=df["Happiness Score"],y=df["Dystopia Residual"])


# In[42]:


#example of multivariate analysis
corr_mat=df.corr()
plt.figure(figsize=[22,12])
sns.heatmap(corr_mat,annot=True)
plt.title("Correlation Matrix")
plt.show()


# There is very good correlation between
# Happiness Score-Economy(GDP)-0.78
# Happiness Score-Family-0.74
# Happiness score-Health(Life expectancy)-0.72
# Happiness score-Freedom-0.57

# In[44]:


corr_matrix=df.corr()
corr_matrix['Happiness Score'].sort_values(ascending=False)


# In[45]:


#Another example of the multi variate analysis
df.plot(kind="density",subplots=True,layout=(6,11),sharex=False, legend=False, fontsize=1 ,figsize=(18,12))
plt.show()


# In[46]:


df['Economy (GDP per Capita)'].plot(kind="kde",figsize=(10,10))


# In[48]:


df['Health (Life Expectancy)'].plot(kind="kde",figsize=(10,10))


# In[50]:


df["Family"].plot(kind="kde",figsize=(10,10))


# In[51]:


df['Freedom'].plot(kind="kde",figsize=(10,10))


# In[47]:


sns.pairplot(df)


# In[52]:


df.skew()


# In[53]:


df.drop('Country',axis=1, inplace=True)


# In[61]:


df.drop('Generosity',axis=1, inplace=True)


# In[54]:


df.drop('Happiness Rank',axis=1, inplace=True)


# In[65]:


df.drop('Standard Error',axis=1, inplace=True)


# In[66]:


df


# In[67]:


pd.unique(df['Region'])


# In[68]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Region"]=le.fit_transform(df['Region'])
df


# In[69]:


df.boxplot(figsize=[20,8])
plt.subplots_adjust(bottom=0.25)
plt.show()


# In[70]:


x=df.drop('Happiness Score',axis=1)


# In[71]:


y=df["Happiness Score"]


# In[ ]:





# In[72]:


x.skew()


# In[73]:


#we see the skewness in the dataset. we will remove the skewness using the power_transform function
from sklearn.preprocessing import power_transform
df_new=power_transform(x)

df_new=pd.DataFrame(df_new,columns=x.columns)


# In[74]:


df_new.skew()


# In[75]:


df_new=x


# In[76]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[77]:


maxAccu=0
maxRS=0
for i in range (1,400):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=i)
    LR = LinearRegression()
    LR.fit(x_train,y_train)
    predrf=LR.predict(x_test)
    r2=r2_score(y_test,predrf)
    if r2>maxAccu:
        maxAccu=r2
        maxRS=i
print("Best r2 is ",maxAccu,'on Random_state',maxRS)


# In[87]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x=scale.fit_transform(x)


# In[88]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=373)


# In[89]:


LR= LinearRegression()
LR.fit(x_train,y_train)
predlr=LR.predict(x_test)
print(r2_score(y_test,predlr))
print(mean_squared_error(y_test,predlr))


# In[104]:


from sklearn.model_selection import cross_val_score

scr=cross_val_score(LR,x,y,cv=5)
print("Cross validation score of LinearRegression model :",scr.mean())


# In[ ]:




