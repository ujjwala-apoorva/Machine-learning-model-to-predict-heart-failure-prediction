#!/usr/bin/env python
# coding: utf-8

# # Machine learning model to predict heart failure prediction

# ### Importing Libraries

# We need to import these modules , so that we can use and call the required function within them and run our program .
# The modules used here are:
# 
# numpy module - for arrays
# 
# pandas module - managing the data sets
# 
# various modules from sklearn- for using various prediction models, for examle: LogisticRegression, KNeighboursClassifier, GridSearchCV.
# 
# Various modules from sklearn- for calculating the accuracy, classification report and confusion matrix.
# 
# plotly module- it is used for ploting different types of graphs and charts in our pipeline.
# 

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt


# In[2]:


from sklearn.metrics import r2_score,classification_report,f1_score,matthews_corrcoef,recall_score
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,mean_squared_error,mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Loading the data sets

# Here we are loading the data set, which is a csv (comma seperatd value) file. We will load using the read_csv function from pandas module.

# In[6]:


#Loading the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

#Print the first 5 rows of the dataframe.
data.tail(13)


# ### Identifying the dependent variable

# In[4]:


#dependent variable
X = data.drop('DEATH_EVENT',axis=1)
y = data['DEATH_EVENT']


# ### Checking the shape of the data set

# In[5]:


#lets check the shape 
X.shape


# ## Data Visualization

# Here we would be using the plotly.figure factory module to create a age distribution graph.

# In[6]:


# age distribution

hist_data =[data["age"].values]
group_labels = ['age'] 

fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text='Age Distribution plot')

fig.show()


# Here in the plot above, we see that:
#     Age wise 40 to 80 the spread is High and in the ages 
#     less than 40 and higher than 80 people are very low.
# 

# ### Gender and age wise comparison

# In[7]:


fig = px.box(data, x='sex', y='age', points="all")
fig.update_layout(
    title_text="Gender wise Age Spread - Male = 1 Female =0")
fig.show()


# ### Heat Map

# In[8]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), vmin=-1, cmap='coolwarm', annot=True);


# ## Data Modeling

# Here we will split our data into training and testing data set respectively.

# A training data set will be used to train our models, and the testing data set will you used to predict the outcomes of our model, after fitting and training of it. The training model is generally larger data set, as we need to train our model. Our test dataset is generally smallaer and values are different from the training data set

# In[9]:


# Train test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# Here I am creating an empty list named accuracy_list, so that we can append the list with the different accuracy values, and in the end get the best model for our data set.

# In[10]:


accuracy_list=[]


# ## Logistic Regression Model

# Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more  independent variables.

# In[11]:


#call the function and give a name
log_reg = LogisticRegression()


# In[12]:


#fitting of the model
log_reg.fit(X_train,y_train)


# In[13]:


#predicting the model
logr_pred = log_reg.predict(X_test)


# In[14]:


#checking the accuracy
accuracyLR = accuracy_score(y_test,logr_pred)


accuracyLR


# In[15]:


accuracy_list.append(100*accuracyLR) #storing the accuracy in accuracy_list list variable we defined before. 


# In[16]:


log_reg = LogisticRegression()
grid = {"penalty" : ["l1", "l2"],"C" : np.arange(0,100,1)}
log_reg_cv = GridSearchCV(log_reg, grid, cv=3)
log_reg_cv.fit(X_train,y_train)


# In[17]:


print("Tuned hyperparameter n_estimators: {}".format(log_reg_cv.best_params_)) 
print("Best score: {}".format(log_reg_cv.best_score_))
print("Best Estimator: {}".format(log_reg_cv.best_estimator_))


# In[18]:


#results
results_NB = pd.DataFrame(log_reg_cv.cv_results_['params'])
results_NB['test_score'] = log_reg_cv.cv_results_['mean_test_score']
results_NB


# In[19]:


#Performance Comparison for Logistics Regression
import matplotlib.pyplot as plt
for i in ['l1', 'l2']:
    temp = results_NB[results_NB['penalty'] == i]
    temp_average = temp.groupby('C').agg({'test_score': 'mean'})
    plt.plot(temp_average, marker = '.', label = i)
    
    
plt.legend()
plt.xlabel('C')
plt.ylabel("Mean CV Score")
plt.title("Logistic Regression Performance Comparison")
plt.show()


# Although Logistic Regression is a simple model, we still tried to do the hyper parameter tuning to see if we can get better results.

# ### Priniting the final results, accuracy , mean absolute error and mean squared error.

# In[20]:


model_LR = log_reg_cv.best_estimator_
model_LR.fit(X_train,y_train)
predictions_LR =  model_LR.predict(X_test)
print('\n')
print('Accuracy: ', accuracy_score(y_test,predictions_LR))
print('f1-score:', f1_score(y_test, predictions_LR))
print('Precision score: ', precision_score(y_test,predictions_LR))
print('Recall score: ', recall_score(y_test,predictions_LR))
print('MCC: ',matthews_corrcoef(y_test,predictions_LR) )
print('Mean Squared Error:', mean_squared_error(y_test, predictions_LR) ** 0.5)
print('Mean Absolute Error:', mean_absolute_error(y_test, predictions_LR) ** 0.5)
print('\n')
print(classification_report(y_test, predictions_LR))
print('\n')
ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test, predictions_LR), annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')


# We saw that the Logistic Regression Model gave quite a high precision and accuracy. The recall value was also close to 1, Thus this model could beconsidered as one of the predicting models for this data set. But we will still check the other models and try to get which would be the best predicting model. In the end we got 84% accuracy which is quite high. The f1 score is also close to 1 , but the reacll value is not very high. Thus we might not choose this model as our best model.

# ## KNNeighbours Classifier Model

# K nearest neighbour classifier model, is also a commonly used model, for both classification and regression types of sata sets. The only drawbacks of this model is, that is takes sime tme to load, and can get a bit slow at times, while haldiling large data sets. This Type of model is perfect for when one does not cares about the time taken to train the model, untill and unless it gives the correct results.

# In[21]:


# k nearest 

knn_parameters = {'n_neighbors' : [i for i in range(1, 40)]}

grid_search_knn = GridSearchCV(estimator = KNeighborsClassifier(), 
                           param_grid = knn_parameters,
                           cv = 10,
                           n_jobs = -1)

grid_search_knn.fit(X_train, y_train)

knn = grid_search_knn.best_estimator_

y_pred_knn = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)

knn_accuracy


# Here we see, that the accuracy is very low than the Logistic Regression model. Thus it makes us sure that we would not be using this model as our predicting model for our data set.

# Still, we will once check the best accuracy score over a range of different neighbours, to see the best fit. We will also plot it on a graph for better visualization

# In[22]:


e_knn=np.zeros(100);
for i in range(0,len(e_knn)):
    knn_model = KNeighborsClassifier(n_neighbors=i+1)
    knn_model.fit(X_train, y_train)
    yh_knn=knn_model.predict(X_test)
    e_knn[i]=accuracy_score(y_test, yh_knn)
    
print("KNN Prediction Accuracy Score: ",np.round(e_knn.max(),3),' with N = ',e_knn.argmax()+1)
plt.plot(np.arange(1,101),e_knn)
plt.plot(e_knn.argmax()+1,e_knn.max(),'or')
plt.title('KNN Accuracy Score')
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.show()


# Here we see that the heighest accuracy is 76%, where the N value is 26.

# In[23]:


accuracy_list.append(100*knn_accuracy) #saving the accuracy into the accuracy list


# ## Random Forest Classifier

# Random forests is a supervised learning algorithm. It can also be used both for classification and regression. It is also the most flexible and easy to use algorithm.Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting. It also provides a pretty good indicator of the feature importance.

# In[24]:


#random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy: ", accuracy_score(y_test, predictions))


# We got a very high accuracy from Random Forest Classifier. Now we will try to do hyperparameter tuning on this model, to see if we can get better results

# ### Hyperparameter Tuning for Random Forest Classifier

# In[25]:


#hyper parameter tuning and Grid Search
grid = {"n_estimators" : np.arange(0,200,2)}
rf = RandomForestClassifier()
rf_random = GridSearchCV(rf, grid, cv=3)
rf_random.fit(X_train,y_train)


# In[26]:


#printing values

print(rf_random.best_params_)
print(rf_random.best_estimator_)


# In[27]:


#Printing Results
results_NB = pd.DataFrame(rf_random.cv_results_['params'])
results_NB['test_score'] = rf_random.cv_results_['mean_test_score']
results_NB


# In[28]:


#NB Performance Comparison 
plt.plot(results_NB['n_estimators'], results_NB['test_score'], marker = '.') 
plt.xlabel('n_estimators')
plt.ylabel("Mean CV Score")
plt.title("NB Performance Comparison")
plt.show()


# ### Comparing the Accuracy, f1 score, recall, and mean squared and absolute error

# In[29]:


model_RF = rf_random.best_estimator_
model_RF.fit(X_train,y_train)
predictions_RF =  model_RF.predict(X_test)
print('\n')
print('Accuracy: ', accuracy_score(y_test,predictions_RF))
print('f1-score:', f1_score(y_test, predictions_RF))
print('Precision score: ', precision_score(y_test,predictions_RF))
print('Recall score: ', recall_score(y_test,predictions_RF))
print('MCC: ',matthews_corrcoef(y_test,predictions_RF) )
print('Mean Squared Error:', mean_squared_error(y_test, predictions_RF) ** 0.5)
print('Mean Absolute Error:', mean_absolute_error(y_test, predictions_RF) ** 0.5)
print('\n')
print(classification_report(y_test, predictions_RF))
print('\n')
ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test, predictions_RF), annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')


# In[30]:


accuracy_list.append(100*accuracy_score(y_test,predictions_RF))


# Here, after the hyperparameter tuning and grid search, we see that the accuracy is 92%, and the recall and f1 score are also very close to 1,of around 0.93 and 0.98 which is a very high score, as it is very difficult to get both precision and recall close to 1. Looking at this confusion matrix and data analysis, we might conclude that yes, Random Forest classifier could be our best fit model for this data set of heart failure prediction

# ## Comparing the best model

# In[31]:


model_list = ['Logistic Regression','KNearestNeighbours', 'RandomForest'] # creating the lis of models used
model_list


# In[32]:


plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# Creating a table for the graph printed above

# In[33]:


models = [('Random Forest Classifier', accuracy_score(y_test,predictions_RF)), 
          ('K-Nearest Neighbour', knn_accuracy),('Logistic Regression',accuracyLR)]

model_comparasion = pd.DataFrame(models, columns=['Model', 'Accuracy Score'])

model_comparasion.head()


# Here both of the graph and in the table , we see that Random Forest Classifier would be the best model fit for our data set to predict the heart failure prediction data set.

# In[ ]:




