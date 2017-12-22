from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import sys


class Evaluator():
    def __init__(self, y_true, y_pred, y_pred_probs):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_probs = y_pred_probs


    def eva_metrics(self):
        '''
        Given a classifier, evaluate by various metrics

        Input:
            y_true: a Pandas dataframe of actual label value
            y_pred: a Pandas dataframe of predicted label value
            y_pred_probs: a Pandas dataframe of probability estimates

        Output:
            rv: a dictionary where key is the metric and value is the score

        '''
        rv = {}
        metrics = {'accuracy': accuracy_score, 'f1_score': f1_score,
                    'precision': precision_score, 'recall': recall_score,
                    'auc': roc_auc_score}
        for metric, fn in metrics.items():
        	rv[metric] = fn(self.y_true, self.y_pred)

        y_pred_probs_sorted, y_true_sorted = zip(*sorted(zip(self.y_pred_probs, self.y_true), reverse=True))
        levels = [1, 3, 5]
        for k in levels:
        	rv['p_at_'+str(k)+'%'] = precision_at_k(y_true_sorted, y_pred_probs_sorted, k)
        	rv['r_at_'+str(k)+'%'] = recall_at_k(y_true_sorted, y_pred_probs_sorted, k)
        rv['p_at_200'] = precision_at_k(y_true_sorted, y_pred_probs_sorted, 200, False)
        rv['r_at_200'] = recall_at_k(y_true_sorted, y_pred_probs_sorted, 200, False)

        return rv

    def compute_confusion_matrix(self):
        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0
        for idx, p_label in enumerate(self.y_pred):
            if p_label == 1 and self.y_true[idx] == 1:
                true_positive +=1
            elif p_label ==1 and self.y_true[idx] == 0:
                false_positive +=1
            elif p_label == 0 and self.y_true[idx] == 1:
                false_negative += 1
            else:
                true_negative += 1

        return false_positive, false_negative, true_positive, true_negative


def generate_binary_at_k(y_pred_probs, k, pct = True):
    '''
    Transform probability estimates into binary at threshold of k
    '''
    if not pct:
        cutoff_index = k
    else:
        cutoff_index = int(len(y_pred_probs) * (k / 100.0))
    y_pred_binary = [1 if x < cutoff_index else 0 for x in range(len(y_pred_probs))]
    return y_pred_binary

def precision_at_k(y_true, y_pred_probs, k, pct=True):
    '''
    Calculate precision score for probability estimates at threshold of k
    '''

    preds_at_k = generate_binary_at_k(y_pred_probs, k, pct)
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_pred_probs, k, pct=True):
    '''
    Calculate recall score for probability estimates at threshold of k
    '''

    preds_at_k = generate_binary_at_k(y_pred_probs, k, pct)
    recall = recall_score(y_true, preds_at_k)
    return recall




def plot_precision_recall_n(y_true, y_pred_probs, model_name):
    '''
    '''

    from sklearn.metrics import precision_recall_curve
    y_score = y_pred_probs
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    plt.figure()
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')

    plt.title(model_name)
    plt.savefig('evaluation/'+model_name)
    plt.close()






