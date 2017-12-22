from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier 
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuit, RandomizedLogisticRegression
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
from .evaluation import Evaluator

#conn = setup_environment.set_connection('2. Pipeline/config.json')

class Model():
    def __init__(self, model_name, model_params, label, cols_to_use, training_data,
                 testing_data):
        self.model_name = model_name
        self.model_params = model_params
        self.label = label
        self.cols_to_use= cols_to_use
        self.training_data = training_data
        self.testing_data = testing_data        

    def get_data(self, df, undersample = True):
        # print warnings for expected columns not found in data
        for col in self.cols_to_use:
            if col not in (list(df.columns.values)):
                print ("column not in {}.format(col)")
                df[col] = 0
        # partition class labels for undersampling
        positive_label_df = df[df[self.label] ==1 ]
        negative_label_df = df[(df[self.label] ==0) & (df[self.label] != None)]

        if undersample:
            len_for_negative_labels =  len(positive_label_df) * 9
            if len_for_negative_labels <= len(negative_label_df):
                negative_label_df = negative_label_df.sample(len_for_negative_labels)

        df = negative_label_df.append(positive_label_df)


        _x = df[self.cols_to_use]

        _y = df[self.label]
        _ids = df['mni_no']
        return _x, _y, _ids

    def get_training_data(self):
        training_x, training_y, training_ids = self.get_data(self.training_data,
                                                             undersample= True)
        return training_x, training_y, training_ids

    def get_test_data(self):
        test_x, test_y, test_ids = self.get_data(self.testing_data)
        return test_x, test_y, test_ids

    def run(self):
        training_x, training_y, training_ids = self.get_training_data()
        test_x, test_y, test_ids = self.get_test_data()
        clf = self.define_model(self.model_name, self.model_params)
        clf.fit(training_x, training_y)
        y_pred = clf.predict(test_x)
        if self.model_name == "linear.SVC":
            y_pred_prob = list(clf.decision_function(test_x))
        else:
            y_pred_prob = list(clf.predict_proba(test_x)[:,1])
        result_dictionary = {
                             'prob_prediction_test_y': list(y_pred_prob),                        
                             'model_name': self.model_name,
                             'model_params': self.model_params,
                             'label': self.label,
                             'feature_columns': self.cols_to_use,
                             'columned_used_for_feat_importance': list(training_x.columns.values)}
        evaluator = Evaluator(test_y, y_pred, y_pred_prob)
        result_dictionary.update(evaluator.eva_metrics())     
        return  result_dictionary

    def define_model(self, model, parameters, n_cores = 0):
        clfs = {'RandomForestClassifier': RandomForestClassifier(n_estimators=50, n_jobs=7),
                'LogisticRegression': LogisticRegression(penalty='l1', C=1e5),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3), 
                'AdaBoostClassifier': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
                'linear.SVC': svm.LinearSVC(),
                'GaussianNB': GaussianNB(),
                'svm.SVC': svm.SVC(kernel='linear', probability=True, random_state=0),
                'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=10, n_jobs=7, criterion='entropy'),
                'GradientBoostingClassifier':GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10)
                }

        if model not in clfs:
            raise ConfigError("Unsupported model {}".format(model))

        clf = clfs[model]
        clf.set_params(**parameters)
        return clf

    def get_feature_importance(self,clf, model_name):
        clfs = {'RandomForestClassifier':'feature_importances',
                'LogisticRegression': 'coef',
                'DecisionTreeClassifier': 'feature_importances',
                'KNeighborsClassifier': None,
                'AdaBoostClassifier': 'feature_importances',              
                'linear.SVC': 'coef',
                'GaussianNB': None, 
                'svm.SVC': 'coef',
                'ExtraTreesClassifier': 'feature_importances',
                'GradientBoostingClassifier': 'feature_importances'}

        if clfs[model_name] == 'feature_importances':
            return  list(clf.feature_importances_)
        elif clfs[model_name] == 'coef':
            return  list(clf.coef_.tolist())
        else:
            return None







