#!/usr/bin/env python
# coding: utf-8

# # Project Description
# The dataset contains grades scored by students throughout their university tenure in various courses and their CGPA calculated based on their grades
# Columns Description-  total 43 columns
# -Seat No : The enrolled number of candidate that took the exams
# 
# CGPA : The cumulative GPA based on the four year total grade progress of each candidate . CGPA is a Final Marks -- provided to student.
#  
# · All other columns are course codes in the format AB-XXX where AB are alphabets representing candidates' departments and XXX are numbers where first X represents the year the canditate took exam
# 
#  
# Predict - CGPA of a student based on different grades in four years.
# 
# Dataset Link-
# •  https://github.com/dsrscientist/dataset4
# •  https://github.com/dsrscientist/dataset4/blob/main/Grades.csv
# 

# Importng important libraries

# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.compose import ColumnTransformer


# In[77]:


df=pd.read_csv("Grades.csv")


# In[78]:


df


# As we can see above data set is having 571 rows and 43 columns , CGPA is the target variable and remaining all as features

# In[79]:


df.shape


# In[80]:


df.columns


# In[81]:


df.info()


# Data Set contains two types of data types object and float

# In[82]:


df.isnull().sum()


# As we can see many of the columns has missing values

# Let's work with missing values.
# as we know for object missing values we cam replace with mode and for int and float with mean value

# We have replaced the null values of each column with the mode lest check th null values again

# In[83]:


df["CY-105"].fillna(df["CY-105"].mode()[0],inplace=True)
df["HS-105/12"].fillna(df["HS-105/12"].mode()[0],inplace=True)
df["MT-111"].fillna(df["MT-111"].mode()[0],inplace=True)
df["CS-106"].fillna(df["CS-106"].mode()[0],inplace=True)
df["EL-102"].fillna(df["EL-102"].mode()[0],inplace=True)
df["EE-119"].fillna(df["EE-119"].mode()[0],inplace=True)
df["ME-107"].fillna(df["ME-107"].mode()[0],inplace=True)
df["CS-107"].fillna(df["CS-107"].mode()[0],inplace=True)
df["HS-205/20"].fillna(df["HS-205/20"].mode()[0],inplace=True)
df["EE-222"].fillna(df["EE-222"].mode()[0],inplace=True)
df["MT-224"].fillna(df["MT-224"].mode()[0],inplace=True)
df["CS-210"].fillna(df["CS-210"].mode()[0],inplace=True)
df["CS-211"].fillna(df["CS-211"].mode()[0],inplace=True)
df["CS-203"].fillna(df["CS-203"].mode()[0],inplace=True)
df["CS-214"].fillna(df["CS-214"].mode()[0],inplace=True)
df["EE-217"].fillna(df["EE-217"].mode()[0],inplace=True)
df["CS-212"].fillna(df["CS-212"].mode()[0],inplace=True)
df["CS-215"].fillna(df["CS-215"].mode()[0],inplace=True)
df["MT-331"].fillna(df["MT-331"].mode()[0],inplace=True)
df["EF-303"].fillna(df["EF-303"].mode()[0],inplace=True)
df["HS-304"].fillna(df["HS-304"].mode()[0],inplace=True)
df["CS-301"].fillna(df["CS-301"].mode()[0],inplace=True)
df["CS-302"].fillna(df["CS-302"].mode()[0],inplace=True)
df["TC-383"].fillna(df["TC-383"].mode()[0],inplace=True)
df["MT-442"].fillna(df["MT-442"].mode()[0],inplace=True)
df["EL-332"].fillna(df["EL-332"].mode()[0],inplace=True)
df["CS-318"].fillna(df["CS-318"].mode()[0],inplace=True)
df["CS-306"].fillna(df["CS-306"].mode()[0],inplace=True)
df["CS-312"].fillna(df["CS-312"].mode()[0],inplace=True)
df["CS-317"].fillna(df["CS-317"].mode()[0],inplace=True)
df["CS-403"].fillna(df["CS-403"].mode()[0],inplace=True)
df["CS-421"].fillna(df["CS-421"].mode()[0],inplace=True)
df["CS-406"].fillna(df["CS-406"].mode()[0],inplace=True)
df["CS-414"].fillna(df["CS-414"].mode()[0],inplace=True)
df["CS-419"].fillna(df["CS-419"].mode()[0],inplace=True)
df["CS-423"].fillna(df["CS-423"].mode()[0],inplace=True)
df["CS-412"].fillna(df["CS-412"].mode()[0],inplace=True)
df["MT-222"].fillna(df["MT-222"].mode()[0],inplace=True)


# In[84]:


#We have replaced the null values of each column with the mode lest check th null values again


# In[85]:


df.isnull().sum()


# Here we can see that there is no null value present in the dataset

# In[86]:


df


# In[87]:


df.value_counts


# In[88]:


df.duplicated().sum()


# There is no duplicate value present in the data set

# In[89]:


sns.heatmap(df.isnull())


# In[90]:


df.describe()


# In[91]:


numerical_col=df.select_dtypes(include=[np.number]).columns
categorical_col=df.select_dtypes(exclude=[np.number]).columns


# In[92]:


categorical_col


# In[93]:


sns.pairplot(data=df,palette="Dark2")


# In[94]:


sns.catplot(data=df, x= "PH-121",y="CGPA")


# In[95]:


sns.catplot(data=df, x= "HS-101",y="CGPA")


# In[96]:


sns.catplot(data=df , x="CY-105",y="CGPA")


# In[97]:


sns.boxplot(x=df["CGPA"])


# We canot remove outliers from target variable

# encoding the object to numerical

# In[98]:


OE=OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# In[99]:


x=df.drop("CGPA",axis =1)
y=df["CGPA"]


# In[100]:


x.shape


# In[101]:


y.shape


# In[102]:


scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
x


# In[103]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif["VIF Values"]=[variance_inflation_factor(x.values,i)for i in range(len(x.columns))]
vif["Features"]= x.columns                


# In[104]:


vif


# Non of the columns are having vif value more than 10

# In[105]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso,Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor


# Modelling

# In[106]:


maxAccu=0
maxRS=0
for i in range(1,300):
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.30,random_state=i)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    acc=r2_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print("Maximum r2 score is ", maxAccu,"on Random_state ",maxRS)


# Linear Regression

# In[107]:


LR=LinearRegression()
LR.fit(x_train,y_train)
pred_LR=LR.predict(x_test)
pred_train=LR.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_LR))
print('R2_score on training data :- ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_LR))
print('Mean suared error :- ',mean_squared_error(y_test,pred_LR))
print('Root mean suared error : - ', np.sqrt(mean_squared_error(y_test,pred_LR)))


# RandomForestregression

# In[108]:


RFR=RandomForestRegressor()
RFR.fit(x_train,y_train)
pred_RFR=RFR.predict(x_test)
pred_train=RFR.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_LR))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_RFR))
print('Mean suared error : -',mean_squared_error(y_test,pred_RFR))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_RFR)))


# In[109]:


# K NeighborRegressor

knn=KNN()
knn.fit(x_train,y_train)
pred_knn=knn.predict(x_test)
pred_train=RFR.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_knn))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_knn))
print('Mean suared error : -',mean_squared_error(y_test,pred_knn))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_knn)))


# In[110]:


# GradientBoostingRegressor

GBR=GradientBoostingRegressor()
GBR.fit(x_train,y_train)
pred_GBR=GBR.predict(x_test)
pred_train=GBR.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_GBR))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_GBR))
print('Mean suared error : -',mean_squared_error(y_test,pred_GBR))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_GBR)))


# In[111]:


#lASSO REGRESSOR

lasso=Lasso()
lasso.fit(x_train,y_train)
pred_lasso=lasso.predict(x_test)
pred_train=lasso.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_lasso))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_lasso))
print('Mean suared error : -',mean_squared_error(y_test,pred_lasso))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_lasso)))


# In[112]:


# Ridge Regressor

rd=Ridge()
rd.fit(x_train,y_train)
pred_rd=rd.predict(x_test)
pred_train=rd.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_rd))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_rd))
print('Mean suared error : -',mean_squared_error(y_test,pred_rd))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_rd)))


# In[113]:


# DecissionTreeRegressor

dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
pred_dtr=dtr.predict(x_test)
pred_train=dtr.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_dtr))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_dtr))
print('Mean suared error : -',mean_squared_error(y_test,pred_dtr))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_dtr)))


# In[114]:


svr=SVR()
svr.fit(x_train,y_train)
pred_svr=svr.predict(x_test)
pred_train=svr.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_svr))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_svr))
print('Mean suared error : -',mean_squared_error(y_test,pred_svr))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_svr)))


# In[115]:


# ExtraTreeRegressor

rtr=ExtraTreesRegressor()
rtr.fit(x_train,y_train)
pred_rtr=rtr.predict(x_test)
pred_train=rtr.predict(x_train)
print('R2_score :- ',r2_score(y_test,pred_rtr))
print('R2 score on training data : - ',r2_score(y_train,pred_train)*100)
print('Mean absolute error :- ',mean_absolute_error(y_test,pred_rtr))
print('Mean suared error : -',mean_squared_error(y_test,pred_rtr))
print('Root mean squared error :- ', np.sqrt(mean_squared_error(y_test,pred_rtr)))


# In[116]:


from sklearn.model_selection import cross_val_score


# In[117]:


score=cross_val_score(LR,x,y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_LR)-score.mean())*100)


# In[118]:


score1=cross_val_score(RFR,x,y)
print(score1)
print(score1.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_RFR)-score1.mean())*100)


# In[119]:


score2=cross_val_score(knn,x,y)
print(score2)
print(score2.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_knn)-score2.mean())*100)


# In[120]:


score3=cross_val_score(GBR,x,y)
print(score3)
print(score3.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_GBR)-score3.mean())*100)


# In[121]:


score4=cross_val_score(lasso,x,y)
print(score4)
print(score4.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_lasso)-score4.mean())*100)


# In[122]:


score5=cross_val_score(rd,x,y)
print(score5)
print(score5.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_rd)-score5.mean())*100)


# In[123]:


score6=cross_val_score(dtr,x,y)
print(score6)
print(score6.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_dtr)-score6.mean())*100)


# In[124]:


score7=cross_val_score(svr,x,y)
print(score7)
print(score7.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_svr)-score1.mean())*100)


# In[125]:


score8=cross_val_score(rtr,x,y)
print(score8)
print(score8.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(y_test,pred_rtr)-score8.mean())*100)


# from the difference of both R2 and cross validation score we have consluded that GradientBoostingRegressor is the best performing model

# In[126]:


from sklearn.model_selection import GridSearchCV


# In[133]:


parameter={'learning_rate':[0.01,0.02,0.03],'subsample':[0.9,0.5,0.2],'n_estimators':[100,500,1000],'max_depth':[4,6,8]}
grid=GridSearchCV(estimator=GBR,param_grid=parameter,cv=2,n_jobs=-1)
grid.fit(x_train,y_train)


# In[134]:


grid.best_params_


# In[135]:


model=GradientBoostingRegressor(learning_rate=0.03,max_depth=8,n_estimators=500,subsample=0.2)


# In[136]:


model.fit(x_train,y_train)
red=model.predict(x_test)
print('R2 score',r2_score(y_test,pred))
print("Mean absolute error :",mean_absolute_error(y_test,pred))
print('Mean squared error :-',mean_squared_error(y_test,pred))
print("Root mean squared error :-",np.sqrt(mean_squared_error(y_test,pred)))


# In[137]:


import pickle
filename="Student grades prediction"
pickle.dump(model,open(filename,'wb'))


# In[138]:


loaded_model=pickle.load(open("Student grades prediction",'rb'))
result=loaded_model.score(x_test,y_test)
print(result*100)


# In[139]:


final=pd.DataFrame([loaded_model.predict(x_test)[:],y_test[:]],index=["Predicted","Original"])
final

