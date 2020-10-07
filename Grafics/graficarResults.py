#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:12:35 2020

@author: daniel
"""

import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sb

def grafic_results(y_real, y_scores, y_pred):

    fpr, tpr, thresholds = roc_curve(y_real, y_scores)
    AUC = auc(fpr, tpr)
    #plt.figure(figsize = (10,10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label = 'AUC: '+ '%.2f' % AUC  )
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    cm_normalize = []
    c_m=confusion_matrix(y_real,y_pred)
    cm_normalize.append(c_m[0]/(len(y_real)/2))
    cm_normalize.append(c_m[1]/(len(y_real)/2))
    cm_normalize = np.vstack(cm_normalize)
    plt.figure()
    heat_map = sb.heatmap(cm_normalize, annot=True)
    
filePath = '/home/daniel/Escritorio/GITA/TSDNewVersion/BERT/resultsProteccion.csv'
results = pd.read_csv(filePath, header = None)

metric = np.asarray(results[1])

index_real = np.where(metric == 'y_real')[0]
index_score = np.where(metric == 'score')[0]
index_pred = np.where(metric == 'y_pred')[0]


y_real_for_experiment = [np.asarray(results.iloc[i,2:]) for i in index_real]
y_pred_for_experiment = [np.asarray(results.iloc[i,2:]) for i in index_pred]
score_for_experiment = [np.asarray(results.iloc[i,2:]) for i in index_score]

y_real_c = [float(value) for value in np.hstack(y_real_for_experiment)]
y_pred_c = [float(value) for value in np.hstack(y_pred_for_experiment)]
score_c = [float(value) for value in np.hstack(score_for_experiment)]

grafic_results(y_real_c, score_c, y_pred_c)
