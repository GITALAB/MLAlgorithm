#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 07:09:51 2020

@author: daniel
"""

import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


def BestGMM_subjects (FeaturesMatrix_list, n_min, n_max, paso = 1, grafic = True, scale = 'normal', covariance = False):
    metrics_n = {}
    if scale == 'normal':
        ns = range(n_min, n_max+1, paso)
    else:
        ns = [2**x for x in range(1,11) if 2**x >= n_min and 2**x < n_max]
    for n in ns:
        print('Components number', n)
        bic_list, aic_list = GMM_subjects(FeaturesMatrix_list,n,Optimizer_mode = True, covariance = covariance)
        metrics_n[n] = [np.mean(bic_list), np.std(bic_list), np.mean(aic_list), np.std(aic_list)]
    
    best_bic = [n_min, metrics_n[n_min][0], metrics_n[n_min][1]]
    best_aic = [n_min, metrics_n[n_min][2], metrics_n[n_min][3]]
    
    for key, value in metrics_n.items():
        if value[0] <= best_bic[1]:
            best_bic = [key, value[0], value[1]]
        if value[2] <= best_aic[1]:
            best_aic = [key, value[2], value[3]]
    
    if grafic:
        components = metrics_n.keys()
        bic_mean = [value[0] for key, value in metrics_n.items()]
        error_bic = [value[1] for key, value in metrics_n.items()]
        aic_mean = [value[2] for key, value in metrics_n.items()]
        error_aic = [value[3] for key, value in metrics_n.items()]
        plt.figure()
        plt.errorbar(components, bic_mean, error_bic, label = 'BIC')
        plt.errorbar(components, aic_mean, error_aic, label = 'AIC')
        plt.legend()
        plt.grid()
        
    return best_bic, best_aic, metrics_n

        
        
def GMM_subjects (FeatureMatrix_list, n, Optimizer_mode = True, covariance = True):
    bic = []
    aic =[]
    features = []
    with tqdm(total = 100, file = sys.stdout) as pbar:
        step = 100/len(FeatureMatrix_list)
        for i,features_subject in enumerate(FeatureMatrix_list):
            gmm_subject = mixture.GaussianMixture(n_components=n, covariance_type='diag',n_init=100)    
            gmm_subject.fit(features_subject)
            means = np.hstack(gmm_subject.means_)
            if covariance:
                covs = np.hstack(gmm_subject.covariances_)
                features.append(np.concatenate((means,covs)))
            else:
                features.append(means)
    
            bic.append(gmm_subject.bic(features_subject))
            aic.append(gmm_subject.aic(features_subject))
            pbar.update(step)

    if Optimizer_mode:
        return bic,aic
    else:
        return features 