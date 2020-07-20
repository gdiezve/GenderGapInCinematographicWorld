#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from warnings import simplefilter
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

def print_scoresClassification(model, X_train, y_train, X_test, y_test, train=True):
    from sklearn.model_selection import cross_validate

    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    if train:
        scores = cross_validate(model, X_train, y_train, cv=10, scoring=scoring)
        ypredTrain = model.predict(X_train)
        Acc_train = scores['test_acc'].mean()
        Precision_train = scores['test_prec_macro'].mean()
        Recall_train = scores['test_rec_macro'].mean()
        F1_train = scores['test_f1_macro'].mean()
        conf_matrix_train = confusion_matrix(y_train, ypredTrain)
    
        print("Train Results:\n===========================================")
        print(f"CV - Accuracy : {Acc_train:.4f}\n")
        print(f"CV -Precision: {Precision_train:.4f}\n")
        print(f"CV -Recall: {Recall_train:.4f}\n")
        print(f"CV -F1 score: {F1_train:.4f}\n")    
        print(f"Confusion_matrix:\n {conf_matrix_train}\n")    
        print('Class Report TRAIN\n', classification_report(y_train, ypredTrain))

    elif train==False:
        scores = cross_validate(model, X_test, y_test, cv=10, scoring=scoring)
        ypredtest = model.predict(X_test)
        Acc_test = scores['test_acc'].mean()
        Precision_test = scores['test_prec_macro'].mean()
        Recall_test = scores['test_rec_macro'].mean()
        F1_test = scores['test_f1_macro'].mean()
        conf_matrix_test = confusion_matrix(y_test, ypredtest)
    
        print("Test Results:\n===========================================")
        print(f"CV -Accuracy : {Acc_test:.4f}\n")
        print(f"CV -Precision: {Precision_test:.4f}\n")
        print(f"CV -Recall: {Recall_test:.4f}\n")
        print(f"CV -F1 score: {F1_test:.4f}\n")    
        print(f"Confusion_matrix:\n {conf_matrix_test}\n") 

        print('\nClass Report TEST\n', classification_report(y_test, ypredtest))


# In[ ]:





# In[ ]:





# In[ ]:




