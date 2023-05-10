#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.compose import ColumnTransformer


# In[3]:


data_train=pd.read_csv("termdeposit_train.csv")
data_test=pd.read_csv("termdeposit_test.csv")


# In[4]:


submit=data_test


# In[5]:


len(submit)


# In[6]:


print(len(data_train) , len(data_test))


# In[7]:


data_train


# There are 31647 rows and 17 feature columns and 1 target variable present in the data set

# In[8]:


data_train.info()


# there are two types of data set present in the data set
# 

# In[21]:


data_train.nunique()


# In[ ]:





# In[9]:


# lets check missing values present in the data set

data_train.isnull().sum()


# There is no missing value present in the data set

# In[10]:


# lets check missing values present in the data set
data_test.isna().sum()


# There is no missing value present in the test data also

# In[13]:


data_train.describe()


# There is a huge difference between max and 75% means outlier is present in the data set . in few columns mean is grater then midean mean skewness enlarged to left

# # Data Visualization

# In[17]:


sns.pairplot(data_train)


# In[18]:


sns.catplot(x ="loan", hue ="subscribed",kind ="count", data = data_train)


# In[20]:


sns.violinplot(x ="loan", y ="age", hue ="subscribed",data = data_train, split = True)


# In[26]:


linear_vars = data_train.select_dtypes(include=[np.number]).columns
cat_attribs = list(data_train.select_dtypes(exclude=[np.number]).columns)


# In[27]:


def plot_boxplot(df, ft):
    sns.boxplot(df[ft])


# In[28]:


fig = plt.figure(figsize=(18, 9))

fig, ax = plt.subplots(4, 2, figsize=(20, 20))
for variable, subplot in zip(linear_vars, ax.flatten()):
    sns.boxplot(x=data_train[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[29]:


#define function to remove outliers
def  outliers(df , ft ):
    
    Q1=df[ft].quantile(0.25)
    Q3=df[ft].quantile(0.75)
    IQR=Q3 - Q1
    # Upper bound
    upper = Q3+1.5*IQR
    # Lower bound
    lower = Q1-1.5*IQR

    ls=df.index[(df[ft] < lower) | (df[ft] > upper)]
    return ls 
    


# In[30]:


index_list=[]
for feature in linear_vars :
    index_list.extend(outliers(data_train , feature ))


# In[31]:


def remove(df, ls):
    ls=sorted(set(ls))
    df=df.drop(ls)
    return df


# In[32]:


data_train= remove(data_train , index_list)


# In[33]:


data_train.shape


# In[37]:


data_train.corr()


# In[38]:


linear_vars


# In[40]:


X = data_train.drop(columns = ['ID','subscribed'],axis=1)
y=data_train['subscribed']


# In[41]:


data_test=data_test.drop(columns = ['ID'],axis=1)


# In[42]:


len(data_test)


# In[43]:


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[44]:


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
categorical_features


# In[45]:


from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=7)


# In[47]:


X_test


# In[48]:


pipeline_lr=Pipeline([("preprocessor",preprocessor),
                     ("lr_reg",LogisticRegression())])

#pipline for Decision Tree Regressor
pipeline_dt=Pipeline([("preprocessor",preprocessor),
                     ("dt_reg",DecisionTreeClassifier())])

#pipline for Random Forest Regressor/
pipeline_rf=Pipeline([("preprocessor",preprocessor),
                     ("rf_reg",RandomForestClassifier())])


#pipline for KNeighbors Regressor
pipeline_kn=Pipeline([("preprocessor",preprocessor),
                     ("rf_reg",KNeighborsClassifier())])

#pipline for Suport Vector  Regressor
pipeline_svm=Pipeline([("preprocessor",preprocessor),
                     ("svm_reg",SVC())])


# In[49]:


pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_svm ]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LogicticRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "Support Vector"}


# In[50]:


for pipe in pipelines:
    pipe.fit(X_train, y_train)


# In[53]:


cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=5)
    cv_results_rms.append(cv_score)
    print("%s : %f " % (pipe_dict[i], cv_score.mean()))


# In[54]:


pred = pipeline_lr.predict(X_test)
pipeline_lr.score(X_test,y_test)


# In[55]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
param_grid = {'lr_reg__C': np.logspace(-3,3,7),'lr_reg__solver' : ['newton-cg', 'lbfgs', 'liblinear'],'lr_reg__penalty':['l1','l2']}
grid_search = GridSearchCV(pipeline_lr,param_grid,cv = 10, scoring = 'neg_mean_squared_error',return_train_score = True)
grid_search.fit(X_train,y_train)


# In[ ]:


search_score = cross_val_score(grid_search,
                               X_train,
                               y_train,
                               scoring="neg_mean_squared_error",
                               cv=10)
search_rmse_score=np.sqrt(-search_score)
print("Scores: ", search_rmse_score)
print("Mean: ", search_rmse_score.mean())
print("Standard Deviation: ", search_rmse_score.std())


# In[ ]:




