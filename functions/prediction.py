# -*- coding: utf-8 -*-

import numpy as np
import functions.time_functions as tf
from scipy import stats
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from random import randint as rt
from random import seed
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import copy
from sklearn.linear_model import Ridge


#%% preprocessing
def equalize_trace_lengths2(transients, outcome):
    seed(0)
    transients = np.transpose(transients)
    nr_strugg = np.count_nonzero(outcome)
    nr_imm = len(outcome)-nr_strugg
    
    if nr_strugg > nr_imm:
        for_delete = []
        keep = np.where(outcome ==1)[0]
        for_delete = np.random.choice(keep, nr_strugg-nr_imm, replace = False)
        outcome = np.delete(outcome, for_delete)
        transients = np.delete(transients, for_delete, 1)
    
    elif nr_strugg < nr_imm:
        for_delete = []
        keep = np.where(outcome ==0)[0]
        for_delete = np.random.choice(keep, nr_imm-nr_strugg, replace = False)
        outcome = np.delete(outcome, for_delete)
        transients = np.delete(transients, for_delete, 1)
    transients = np.transpose(transients)
    return transients, outcome


    


def equalize_trace_lengths(transients1, transients2, outcome1, outcome2): 
    seed(0)
    if len(transients1) > len(transients2):
        transients1 = transients1[:len(transients2)]
        outcome1 = outcome1[:len(transients2)]

    elif len(transients2) > len(transients1):
        transients2 = transients2[:len(transients1)]
        outcome2 = outcome2[:len(transients1)]
    
    return transients1, transients2, outcome1, outcome2


def equalize_behavior_lengths(transients1, transients2, outcome1, outcome2):
    seed(0)
    
    nr_strugg_1 = np.count_nonzero(outcome1)
    nr_imm_1 = len(outcome1)-nr_strugg_1

    nr_strugg_2 = np.count_nonzero(outcome2)
    nr_imm_2 = len(outcome2)-nr_strugg_2
    
    ## give both equal struggle lengths
    if nr_strugg_1 > nr_strugg_2:
        keep = np.where(outcome1 ==1)[0]
        for_delete = np.random.choice(keep, nr_strugg_1-nr_strugg_2, replace = False, )
        outcome1 = np.delete(outcome1, for_delete)
        transients1 = np.delete(transients1, for_delete, 0)
    elif nr_strugg_2 > nr_strugg_1:
        keep = np.where(outcome2 ==1)[0]
        for_delete = np.random.choice(keep, nr_strugg_2-nr_strugg_1, replace = False)
        outcome2 = np.delete(outcome2, for_delete)
        transients2 = np.delete(transients2, for_delete, 0)
    
    # give both equal immobility length
    if nr_imm_1 > nr_imm_2:
        keep = np.where(outcome1 ==0)[0]
        for_delete = np.random.choice(keep, nr_imm_1-nr_imm_2, replace = False)
        outcome1 = np.delete(outcome1, for_delete)
        transients1 = np.delete(transients1, for_delete, 0)
    
    elif nr_imm_2 > nr_imm_1:
        keep = np.where(outcome2 ==0)[0]
        for_delete = np.random.choice(keep, nr_imm_2-nr_imm_1, replace = False)
        outcome2 = np.delete(outcome2, for_delete)
        transients2 = np.delete(transients2, for_delete, 0)
    
   
    # give both balanced dataset
    nr_strugg = np.count_nonzero(outcome1)
    nr_imm = len(outcome1)-nr_strugg
    
    if nr_strugg > nr_imm:
        keep = np.where(outcome1 ==1)[0]
        for_delete = np.random.choice(keep, nr_strugg-nr_imm, replace = False)

    elif nr_strugg < nr_imm:
        keep = np.where(outcome1 ==0)[0]
        for_delete = np.random.choice(keep, nr_imm-nr_strugg, replace = False)    
    
    outcome1 = np.delete(outcome1, for_delete)
    outcome2 = np.delete(outcome2, for_delete)
    transients1 = np.delete(transients1, for_delete, 0)
    transients2 = np.delete(transients2, for_delete, 0)
    
    
    return transients1, transients2, outcome1, outcome2
    
def preprocess_data(baseline, tst, transients):
    idx = np.any(np.vstack((baseline, tst)), 0)
    outcome = tst[idx]
    transients = transients[:, idx]
    transients, outcome = equalize_trace_lengths2(transients.T, outcome)
    transients = tf.standardize_transients(transients, method='standard')
    return transients, outcome

#%%

def neuron_shuffling_fixed_seed(transients):
    """
    Shuffles the given array of transients using a fixed seed for the random number generator.
    Parameters:
        transients (ndarray): An array of transients.
    Returns:
        ndarray: The shuffled array of transients.
    """
    seeds = np.arange(0, len(transients))
    transients_shuff = copy.deepcopy(transients)
    for i in range(len(transients)):
        np.random.seed(seeds[i])
        randnr = rt(0, len(transients[i])-0, )
        first_trace = transients_shuff[i,0:randnr]
        second_trace = transients_shuff[i,randnr:len(transients[i])]
        transients_shuff[i,:] = np.concatenate((second_trace, first_trace))
    return transients_shuff



## DECODING
def create_model(model_type, C=0.01):
    if model_type == 'LogisticRegression':
        return LogisticRegression(C=C, max_iter=10000000, random_state=0)
    elif model_type == 'GaussianNB':
        return GaussianNB()
    elif model_type == 'RidgeClassifier':
        return RidgeClassifier(alpha=50, random_state=0)
    elif model_type == 'SVC':
        return SVC(C=1, max_iter=10000000, random_state=0, kernel='linear')
    else:
        raise ValueError("Invalid model type provided.")

def calculate_scores(model, transients1, transients2, outcome1, outcome2, kfold):
    scores1, scores2, scores_shuffle_1, scores_shuffle_2 = [], [], [], []
    transients2_shuffle = np.transpose(neuron_shuffling_fixed_seed(np.transpose(transients2)))

    for train_index, test_index in kfold.split(transients1):
        X_train, X_test_1 = transients1[train_index], transients1[test_index]
        y_train, y_test_1 = outcome1[train_index], outcome1[test_index]
        _, X_test_2 = transients2[train_index], transients2[test_index]
        _, y_test_2 = outcome2[train_index], outcome2[test_index]
        _, X_test_shuff = transients2_shuffle[train_index], transients2_shuffle[test_index]

        model.fit(X_train, y_train)
        scores1.append(metrics.accuracy_score(y_test_1, model.predict(X_test_1)))
        scores2.append(metrics.accuracy_score(y_test_2, model.predict(X_test_2)))
        scores_shuffle_1.append(metrics.accuracy_score(y_test_1, model.predict(X_test_shuff)))
        scores_shuffle_2.append(metrics.accuracy_score(y_test_2, model.predict(X_test_shuff)))

    return np.mean(scores1), np.mean(scores2), np.mean(scores_shuffle_1), np.mean(scores_shuffle_2)

def decode_tst_baseline_days(transients1, transients2, tst1, tst2, baseline1, baseline2, model_type='LogisticRegression'):
    seed(0)
    
    transients1, outcome1 = preprocess_data(baseline1, tst1, transients1)
    transients2, outcome2 = preprocess_data(baseline2, tst2, transients2)
    transients1, transients2, outcome1, outcome2 = equalize_trace_lengths(transients1, transients2, outcome1, outcome2)
    transients1, outcome1 = shuffle(transients1, outcome1)
    transients2, outcome2 = shuffle(transients2, outcome2)

    model = create_model(model_type, C=0.1)
    kfold = KFold(n_splits=10, shuffle=False)

    return calculate_scores(model, transients1, transients2, outcome1, outcome2, kfold)



def decode_struggle_immobility(transients, immobility, movement, shuffle=True, mod='LogisticRegression'): 
    seed(0)
    elen = len(transients)
    idx = np.logical_or(immobility, movement)
    outcome = movement[idx]
    transients = transients[:, idx]
    
    
    transients, outcome = equalize_trace_lengths2(transients.T, outcome)

    if elen > 1:
        transients = tf.standardize_transients(transients, method='standard')
    
    folds = 10
    reg = create_model(mod, C=0.01)
    cv = KFold(n_splits=folds, shuffle=False)
    
    predicted = cross_val_predict(reg, transients, outcome, cv=cv)
    score = accuracy_score(outcome, predicted)
    
    shuff_trans = np.transpose(neuron_shuffling_fixed_seed(np.transpose(transients)))
    predicted = cross_val_predict(reg, shuff_trans, outcome, cv=cv)
    shuffle_score = accuracy_score(outcome, predicted)
    
    return score, shuffle_score




def between_binary(transients1, transients2, struggle1, struggle2, immobility1, immobility2, stand_method='minmax', C=0.01): 
    np.random.seed(0) 
    
    idx = np.any(np.vstack((immobility1, struggle1)),0)
    outcome1 = struggle1[idx]
    transients1 = transients1[idx]

    idx = np.any(np.vstack((immobility2, struggle2)),0)
    outcome2 = struggle2[idx]
    transients2 = transients2[idx]
    
    transients1, transients2, outcome1, outcome2 = equalize_behavior_lengths(transients1, transients2, outcome1, outcome2)
    
    
    transients1 = tf.standardize_transients(transients1, method=stand_method)
    transients2 = tf.standardize_transients(transients2, method=stand_method)
    
    # Create model 0.01
    model = LogisticRegression(C=C, max_iter = 100000000, random_state = 0)
    
    # Create a KFold object with 10 folds
    kfold = KFold(n_splits=10, shuffle=False)
    
    # Use a for loop to perform cross-validation
    scores1 = []
    scores2 = []
    
    for train_index, test_index in kfold.split(transients1):
        # Split the data into training and testing sets
        X_train, X_test_1 = transients1[train_index], transients1[test_index]
        y_train, y_test_1 = outcome1[train_index], outcome1[test_index]

        _, X_test_2 = transients2[train_index], transients2[test_index]
        _, y_test_2 = outcome2[train_index], outcome2[test_index]
        
        # Fit the model on the training data and calculate the score on the testing data
        reg = model.fit(X_train, y_train)
        
        # predict transients 1 and 2
        predict1 = reg.predict(X_test_1)  
        predict2 = reg.predict(X_test_2)
        
        # get scores
        score1 = metrics.accuracy_score(y_test_1, predict1)
        score2 = metrics.accuracy_score(y_test_2, predict2)
        
        # Append the score and predicted to the list of scores
        scores1.append(score1)
        scores2.append(score2)

    # calculate the mean of the scores
    mean_score1 = np.mean(scores1)
    mean_score2 = np.mean(scores2)

    return mean_score1, mean_score2


def between_continous(transients1, outcome1, transients2, outcome2, stand_method='minmax'): 
    np.random.seed(0)
    transients1, transients2, outcome1, outcome2 = equalize_trace_lengths(transients1, transients2, outcome1, outcome2)
    
    transients1 = tf.standardize_transients(transients1, method=stand_method)
    transients2 = tf.standardize_transients(transients2, method=stand_method)
    
    # Create model
    model = Ridge(alpha = 50)
    
    # Create a KFold object with 10 folds
    kfold = KFold(n_splits=10, shuffle=False)
    
    # Use a for loop to perform cross-validation
    scores1 = []
    scores2 = []
    predicted1 = []
    predicted2 = []
    
    for train_index, test_index in kfold.split(transients1):
        # Split the data into training and testing sets
        X_train, X_test_1 = transients1[train_index], transients1[test_index]
        y_train, y_test_1 = outcome1[train_index], outcome1[test_index]

        _, X_test_2 = transients2[train_index], transients2[test_index]
        _, y_test_2 = outcome2[train_index], outcome2[test_index]
        
        # Fit the model on the training data and calculate the score on the testing data
        reg = model.fit(X_train, y_train)
        
        # predict transients 1 and 2
        predict1 = reg.predict(X_test_1)  
        predict2 = reg.predict(X_test_2)
        
        # get scores
        score1 = stats.pearsonr(predict1,y_test_1)[0]
        score2 = stats.pearsonr(predict2, y_test_2)[0]
        
        # Append the score and predicted to the list of scores
        scores1.append(score1)
        scores2.append(score2)
        predicted1.append(predict1)
        predicted2.append(predict2)
    
    # calculate the mean of the scores
    mean_score1 = np.mean(scores1)
    mean_score2 = np.mean(scores2)
    
    predicted1 = np.hstack(predicted1)
    predicted2 = np.hstack(predicted2)
    
    return mean_score1, mean_score2, predicted2, outcome2 , predicted1, outcome1
        


