# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
import copy
from scipy import signal


def standardize_transients(transients, method='standard'):
   if method == 'standard':
       scaler = preprocessing.StandardScaler().fit(transients)
   elif method == 'quantile':
       scaler = preprocessing.QuantileTransformer(n_quantiles=100).fit(transients)
   elif method == 'minmax':
       scaler = preprocessing.MinMaxScaler().fit(transients)
   else:
       raise ValueError("Invalid standardization method.")
   
   return scaler.transform(transients)

def convolve(transients, width):
    n = 0
    for neuron in transients:
        spike_train_1_neuron = transients[n,:]
        kernel = signal.windows.gaussian(len(spike_train_1_neuron), width)
        transients[n,:] = signal.fftconvolve(spike_train_1_neuron, kernel, mode='same')
        n+=1
    return transients 


    
def remove_edge_behavior(behavior):
    behavior_transform = copy.deepcopy(behavior)
    if behavior[0] == 1:
        e = 0
        while behavior[e] ==1:
            behavior_transform[e] = 0
            e+=1
    if behavior[-1] == 1:
        e = 1
        while behavior[-e] ==1:
            behavior_transform[-e] = 0
            e+=1
    return behavior_transform 


def start_end(behavior, min_frames, max_frames, cut_frames):
    start,end=[],[]
    start_i = 0
    for i in range(1,len(behavior)-1):
        if behavior[i] == 1 and behavior[i-1] == 0:
            start_i = i+cut_frames
        if behavior[i] == 1 and behavior[i+1] == 0:
            end_i = i+1
            if max_frames > end_i-start_i > min_frames:
                start.append(start_i)
                end.append(end_i)
    return start, end


def get_relative_time(transients, start, end):
    rel_time, transients_time = [], []
    for i in range(len(start)):
        epoch_bin = np.linspace(0,1,end[i]-start[i])
        rel_time.extend(epoch_bin)
        transients_time.append(transients[:,start[i]:end[i]])
    transients_time = np.hstack(transients_time)
    return np.array(rel_time), np.array(transients_time)

def make_time(time, min_frames, max_frames = 20000, cut_frames = 0): 
    scaled_time = np.ones(len(time))*-1
    start = 0
    for i in range(1,len(time)-1):
        if time[i] == 1 and time[i-1] == 0:
            start = i+cut_frames
        if time[i] == 1 and time[i+1] == 0:
            stop = i+1
            if max_frames > stop-start > min_frames:
                epoch = time[start:stop]
                epoch_bin = np.arange(0,len(epoch))
                epoch_bin = epoch_bin.reshape(-1,1)
                min_max_scaler = preprocessing.MinMaxScaler()
                scaled_epoch = min_max_scaler.fit_transform(epoch_bin)[:,0]
                scaled_time[start:stop] = scaled_epoch
    return scaled_time

def combine_time(time1,time2):
    for i in range(len(time1)):
        if time1[i] == -1:
            time1[i] = np.nan
            
    for i in range(len(time2)):
        if time2[i] == -1:
            time2[i] = np.nan
        else:
            time2[i] = time2[i] *-1
            
    color2 = copy.deepcopy(time1)
    for i in range(len(time2)):
        if time2[i] <= 0:
            color2[i] = time2[i]
    return color2



def scale_transients(transients, time):
    time = np.append(time,time[-3])
    for i in range(len(time)-1):
        if time[i] < time[i-1]:
            beg = i
        elif time[i] > time[i+1]:
            trace = transients[:,beg:i+1]
            trace = np.transpose(trace)
            scaler = preprocessing.MinMaxScaler().fit(trace)
            transients[:,beg:i+1] = np.transpose(scaler.transform(trace))

    return transients



