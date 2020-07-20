#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.graph_objs as go
import plotly
import plotly.express as px

from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

import math
import random

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[3]:


def load_data(path):
  # Load the data from data sources
  path_to_file = path
  data = pd.read_csv(path_to_file, encoding='ISO-8859-1')

  return data


# In[4]:


def get_info_dataset(data):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables"""
    
    print('Basic information from your dataset\n','---------------------------------------------')
    data.info()


# In[5]:


def get_info_dataset2(data, bool):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables
    num: Boolean index. if it's True that means that you want to proint two list that
    will contain the col names of the categorical and numerical variables, respectively"""
    
    print('Basic information from your dataset\n','---------------------------------------------')
    data.info()
    
    if bool == True:
       num_var = data.select_dtypes(include=['int', 'float']).columns
       print('Numerical variables are:\n', num_var)
       print('-------------------------------------------------')

       categ_var = data.select_dtypes(include=['category', object]).columns
       print('Categorical variables are:\n', categ_var)
       print('-------------------------------------------------') 


# In[6]:


def get_info_dataset2(data, bool_index):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables
    num: Boolean index. if it's True that means that you want to proint two list that
    will contain the col names of the categorical and numerical variables, respectively"""
    
    
    print('Basic information from your dataset\n','---------------------------------------------')
    data.info()
    
    num_var = object
    categ_var = object
    
    if bool_index == True:
       num_var = data.select_dtypes(include=['int', 'float']).columns
       print('Numerical variables are:\n', num_var)
       print('-------------------------------------------------')

       categ_var = data.select_dtypes(include=['category', object]).columns
       print('Categorical variables are:\n', categ_var)
       print('-------------------------------------------------')
    return num_var, categ_var


# In[7]:


def get_info_datasetPrint(data, bool_index, print_index):
    """ This function will be used to extract info from the dataset
    input: dataframe containing all variables
            bool_index: Boolean index. if it's True that means that you want to proint two list that
            will contain the col names of the categorical and numerical variables, respectively
            print_index : True / False . If True == will show the results on the python screen"""
    
    if bool_index == True:
        num_var = data.select_dtypes(include=['int', 'float']).columns
        categ_var = data.select_dtypes(include=['category', object]).columns
        
    if print_index == True:
        print('Numerical variables are:\n', num_var)
        print('-------------------------------------------------')

        print('Categorical variables are:\n', categ_var)
        print('-------------------------------------------------')
    return num_var, categ_var


# In[8]:


def percentage_nullValues(data):
    """
    Function that calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    """
    null_perc = round(data.isnull().sum() / data.shape[0],3) * 100.00
    null_perc = pd.DataFrame(null_perc, columns=['Percentage_NaN'])
    null_perc= null_perc.sort_values(by = ['Percentage_NaN'], ascending = False)
    return null_perc


# In[ ]:


def corrValues(data, var):
    """
    Function that returns the most correlated values with the indicated variable
    input: data --> dataframe
           var --> variable/column of interest from the dataframe specified
    """
    corr = data[var].corr()
    corr = pd.DataFrame(corr, columns=['Correlation_values'])
    corr= corr.sort_values(by = ['Correlation_values'], ascending = False)
    return corr


# In[9]:


def select_threshold(data, thr):
    """
    Function that  calculates the percentage of missing values in every column of your dataset
    input: data --> dataframe
    
    """
    null_perc = percentage_nullValues(data)
      
    col_keep = null_perc[null_perc['Percentage_NaN'] <thr]
    col_keep = list(col_keep.index)
    print('Columns to keep:',len(col_keep))
    print('Those columns have a percentage of NaN less than', str(thr), ':')
    print(col_keep)
    data_c= data[col_keep]
    
    return data_c


# In[2]:


def corrValues(data, var, thr):
    """
    Function that returns a dataframe including the most correlated variables with 
    the indicated one in the parameters
    input: data --> dataframe
           var --> variable/column of interest from the dataframe specified
           thr --> threshold 
    """
    corr = abs(data.corr())
    corr = corr[var].sort_values(ascending = False).to_frame()
    corr = corr.loc[corr.SalePrice > thr, :]

    cols = []
    for i in corr.index:
        cols.append(i)

    return data[cols]


# In[10]:


def fill_na(data):
    """
    Function to fill NaN with mode (categorical variabls) and mean (numerical variables)
    input: data -> df
    """
    for column in data:
        if data[column].dtype != 'object':
            data[column] = data[column].fillna(data[column].mean())  
        else:
            data[column] = data[column].fillna(data[column].mode()[0]) 
    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data


# In[11]:


# Cool way: calling a function inside your function
def fill_naCool(data):
    """
    Function to fill NaN with mode (categorical variabls) and mean (numerical variables)
    input: data -> df
    """
    num_var, categ_var = get_info_datasetPrint(data, True, False)
    
    data[num_var] = data[num_var].fillna(data[num_var].mean())
    data[categ_var] = data[categ_var].fillna(data[categ_var].mode()[0]) 

    print('Number of missing values on your dataset are')
    print()
    print(data.isnull().sum())
    return data


# In[34]:


def corrCoef(data):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    
    input: data->dataframe        
    """
    num_var, categ_var = get_info_datasetPrint(data, True, False)
    data_num = data[num_var]
    data_corr = data_num.corr()
    
    mask = np.zeros_like(data_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(15, 10))
    sns.heatmap(data_corr,
                xticklabels = data_corr.columns.values,
               yticklabels = data_corr.columns.values,
                mask = mask,
               annot = True, vmax=1, vmin=-1, center=0, cmap= 'viridis')


# In[1]:


def corrCoef_Threshold(data, threshold):
    """
    Function aimed to calculate the corrCoef between each pair of variables
    
    input: data->dataframe 
           threshold -> float value we indicate as threshold to select variables
            
            """
    num_var, categ_var = get_info_datasetPrint(data, True, False)
    data_num = data[num_var]
    data_corr = abs(data_num.corr())
    data_cols = data_corr.columns
    
    data_corr= pd.DataFrame(data_corr.unstack().sort_values(ascending = False), columns = ['corrCoef'])
    # threshold that I want to select. I will keep the variables with a corrCoef higher than the threshols
    thr = float(threshold)
    data_corr = data_corr[(data_corr.corrCoef >thr)].unstack()
    data_corr = pd.DataFrame(data_corr)
    
    mask = np.zeros_like(data_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Create the plot
    plt.figure(figsize=(15, 10))
    sns.heatmap(data_corr,xticklabels = data_cols,
                mask=mask,
                yticklabels = data_cols,
                annot = True, vmax=1, vmin=-1, center=0, cmap= 'viridis')


# In[14]:


def createBoxplot2traces(data, var1, var2):
    
    label_trace1 = str(input('Name for trace 1:'))
    label_trace2 = str(input('Name for trace 2:'))

   # dataframe --> columns  
    trace1 = go.Box(
        y = data[var1],
        name = label_trace1,
        marker = dict(
        color = 'rgb(12, 12, 140)',
        )
    )

    trace2 = go.Box(
        y = data[var2],
        name = label_trace2,
        marker = dict(
        color = 'rgb(12, 128, 128)',
        )
    )
    
    data = [trace1, trace2]
    iplot(data)


# In[15]:


def dropOutliers_IQR(data, var_name):
    """
    Function aimed to remove outliers based on IQR method
    
    input: data->dataframe 
           var_name -> name of feature (column) we want to treat
            
            """
    copy = data.copy()
    variable_select = copy[var_name]
    print('Size before droping outliers:', variable_select.size)
    print()
    from scipy.stats import iqr, skew
    Q1 = copy[var_name].quantile(0.25)
    Q3 = copy[var_name].quantile(0.75)
    IQR = Q3 - Q1

    upperlimit = Q3 + 1.5*IQR
    lowerlimit = Q1 - 1.5*IQR

    copy[var_name] = copy[var_name].loc[(copy[var_name]  < upperlimit) & (copy[var_name]  > lowerlimit)]
    copy = copy.dropna()
    print('Size after dropping outliers:', copy[var_name].size)
    return copy


# In[16]:


def get_Stats(data):
    """
    Function aimed to get min, mean and max values of a dataframe.
    
    input: data->dataframe
            
            """
    statist = []
    for col in data:
        min_var = data[col].min()
        mean_var = data[col].mean()
        max_var = data[col].max()
        list_metrics = [min_var, mean_var, max_var]
        statist.append(list_metrics)
        
    statist = pd.DataFrame(statist,columns = ['min', 'mean', 'max'], index = [data.columns])
    #print(statist)
    return statist


# In[17]:


def boxplot_var(df, variable, title):
    """Function that creates a boxplot
    df --> dataframe
    variable --> feature you want to analyse
    title --> title for the boxplot"""
    boxplot = go.Box(
    x=df[variable],
    name = df[variable].name,
    marker = dict(
        color = 'rgb(12, 12, 140)',
    ),
    orientation='h',
    )
    layout = go.Layout(title=title)
    fig = go.Figure(data=boxplot, layout=layout)
    fig.show()


# In[19]:


def boxplot2(df):
    """Function that creates a boxplot of all variables from the specified dataframe
    df --> dataframe"""

    import randomcolor
    rand_color = randomcolor.RandomColor()
    
    fig = go.Figure(data=[go.Box(
        y= df[i],
        name=i,
        ) for i in df])
    
    fig.for_each_trace(
        lambda trace: trace.update(marker_color=rand_color.generate(format_='rgb',
                                                                    luminosity='bright')[0])
    )

    # format the layout
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
        yaxis=dict(zeroline=False, gridcolor='grey'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        title='Distribution of variables'
    )

    fig.show()


# In[64]:


def boxplot(df):
    """Function that creates a boxplot of all variables from the specified dataframe
    df --> dataframe"""    
    fig = go.Figure()
    count = 0
    
    for i in df:
        fig.add_trace(go.Box(
            y= df[i],
            name=i,
            marker_color=px.colors.sequential.Viridis[count]
        ))
        if count == 9:
            count = 0
        else:
            count += 1
    
    # format the layout
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
        yaxis=dict(zeroline=False, gridcolor='grey'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        title='Distribution of variables'
    )
    
    fig.show()


# In[19]:


def get_Stats(data):
    statist = []
    
    for col in data:
        min_var = data[col].min()
        mean_var = data[col].mean()
        max_var = data[col].max()
        list_metrics = [min_var, mean_var, max_var]
        statist.append(list_metrics)
        
    statist = pd.DataFrame(statist,columns = ['min', 'mean', 'max'], index = [data.columns])
    #print(statist)
    return statist


# In[20]:


def displot(var):
    """Function that creates a histogram / distribution plot
    var --> variable that needs to be plotted (i.e.: data['var'])"""
    trace = go.Histogram(x=var,
                            opacity=0.75,
                            name = var.name,
                            marker_color='darkcyan',)
    layout = go.Layout(title='Distribution of '+ var.name + ' variable',
                   yaxis=dict( title='Count'),)
    fig = go.Figure(data=trace, layout=layout)
    iplot(fig)


# In[66]:


def PlotPie(df, nameOfFeature):
    """
    Pie Chart in order to represent the distribution of each category in each Variable
    Make sure your values starts from 0 if it is numerical
    """
    if [df[nameOfFeature]] != 0:
        labels = [str(df[nameOfFeature].unique()[i]) for i in range(df[nameOfFeature].nunique())]
        values = [df[nameOfFeature].value_counts()[i+1] for i in range(df[nameOfFeature].nunique())]
    else:
        labels = [str(df[nameOfFeature].unique()[i]) for i in range(df[nameOfFeature].nunique())]
        values = [df[nameOfFeature].value_counts()[i] for i in range(df[nameOfFeature].nunique())]
    
    df = px.data.tips()
    fig = px.pie(df, values=values, names=labels, 
                   color_discrete_sequence=px.colors.sequential.Viridis,
                  title='Distribution of '+ nameOfFeature + ' variable')
    
    fig.show()


# In[21]:


def split_data(X, y, test_size, reshape):
    """
    Function created to split the data.
    inputs: X --> non target variables
            y --> target variable
            test_size --> coefficient that indicates the desidered size of test sample
            reshape --> if True, function will return reshaped splits
    output: X_train, X_test, y_train, y_test
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=0)
    
    if reshape == True:
        y_train =  np.array(y_train).reshape(-1, 1) 
        X_train = np.array(X_train).reshape(-1, 1) 

        y_test =  np.array(y_test).reshape(-1, 1)  
        X_test = np.array(X_test).reshape(-1, 1)
        
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test        


# In[22]:


def predict_data(X_train,X_test,model,reshape):
    """
    Function created to preduct the data.
    inputs: X_train --> to predict y_predTrain variable
            X_test --> to predict y_predTest variable
            model --> model used to predict
            reshape --> if True, function will return reshaped splits
    output: y_predTrain, y_predTest
    """
    y_predTrain = model.predict(X_train)
    
    y_predTest = model.predict(X_test)
    
    if reshape == True:
        y_predTrain =  np.array(y_predTrain).reshape(-1, 1)
        
        y_predTest =  np.array(y_predTest).reshape(-1, 1)
        
        return y_predTrain, y_predTest
    else:
        return y_predTrain, y_predTest


# In[23]:


def get_model_metrics(X_train,y_train,X_test,y_test,y_predTrain,y_predTest):
    """
    Function that predicts and returns metrics of a model for train and test datasets.
    inputs: X_train,y_train,X_test,y_test,y_predTrain,y_predTest --> data splited
    """
    
    statist_train = []
    MAE_lTrain = metrics.mean_absolute_error(y_train, y_predTrain)
    MSE_lTrain = metrics.mean_squared_error(y_train,y_predTrain)
    RMSE_lTrain = np.sqrt(metrics.mean_squared_error(y_train, y_predTrain))
    R2_lTrain = r2_score(y_train, y_predTrain)
    train = 'Train'

    list_metrics = [MAE_lTrain, MSE_lTrain, RMSE_lTrain, R2_lTrain, train]
    statist_train.append(list_metrics)
    statist_train = pd.DataFrame(statist_train,columns = ['MAE', 'MSE', 'RMSE', 'R2','Dataset'])
    
    statist_test = []
    MAE = metrics.mean_absolute_error(y_test, y_predTest)
    MSE = metrics.mean_squared_error(y_test, y_predTest)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_predTest))
    R2 = r2_score(y_test, y_predTest)
    test = 'Test'
    
    list_metrics = [MAE, MSE, RMSE, R2, test]
    statist_test.append(list_metrics)
    statist_test = pd.DataFrame(statist_test,columns = ['MAE', 'MSE', 'RMSE', 'R2', 'Dataset'])
    
    statist = pd.merge(statist_train,statist_test, how='outer').set_index('Dataset')
    
    return statist


# In[30]:


def get_scoresClassification(model, X_train, y_train, X_test, y_test,y_predTrain,y_predTest):
    """
    Function that predicts and returns scores of a model for train and test datasets.
    inputs: model --> model we are going to use to predict the scores
            X_train,y_train,X_test,y_test --> data splited
            train --> if True, it will give us the train scores; otherwise, it will 
            return the test ones
    """
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    scores_train = []
    scores = cross_validate(model, X_train, y_train, cv=10, scoring=scoring)
    Acc_train = scores['test_acc'].mean()
    Precision_train = scores['test_prec_macro'].mean()
    Recall_train = scores['test_rec_macro'].mean()
    F1_train = scores['test_f1_macro'].mean()
    conf_matrix_train = confusion_matrix(y_train, y_predTrain)
    train = 'Train'
    
    list_scores = [Acc_train, Precision_train, Recall_train, F1_train, train]
    scores_train.append(list_scores)
    scores_train = pd.DataFrame(scores_train,columns = ['Accuracy', 'Precision', 'Recall', 
                                                          'F1_score', 'Dataset'])
    
    scores_test = []    
    scores = cross_validate(model, X_test, y_test, cv=10, scoring=scoring)
    Acc_test = scores['test_acc'].mean()
    Precision_test = scores['test_prec_macro'].mean()
    Recall_test = scores['test_rec_macro'].mean()
    F1_test = scores['test_f1_macro'].mean()
    conf_matrix_test = confusion_matrix(y_test, y_predTest)
    test = 'Test'
    
    list_scores = [Acc_test, Precision_test, Recall_test, F1_test, test]
    scores_test.append(list_scores)
    scores_test= pd.DataFrame(scores_test,columns = ['Accuracy', 'Precision', 'Recall', 
                                                          'F1_score', 'Dataset'])
    
    scores = pd.merge(scores_train,scores_test, how='outer').set_index('Dataset')
    
    print('Train confusion matrix')
    print('-----------------------')
    print(conf_matrix_train)
    print('')
    print('Test confusion matrix')
    print('-----------------------')
    print(conf_matrix_test)
    
    return scores


# In[25]:


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    """Function that plots the decision boundary for the classifier.
    inputs: clf --> model
            X,Y --> variables we want to analyse
    """
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k')


# In[ ]:




