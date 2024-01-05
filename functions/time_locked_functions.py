# -*- coding: utf-8 -*-


import copy
import numpy as np
from scipy import stats
import functions.rate_coding
from sklearn import preprocessing
from scipy.signal import correlate
from scipy.signal import correlation_lags
import matplotlib.pyplot as plt
#%%



def Time_locked_activity(data, frames, day = 'day1', movement ='tst_speed_dlc', 
                spikes='fn_tst', behavior='immobility_time', seltype = 'struggle', calcium=True, multisess = False, indices_prev = False, sig_seltype = True):
    spike_mean = [None] * len(data)
    spike_sem = [None] * len(data)
    lags = [None] * len(data)
    behavior_mean = [None] * len(data)
    behavior_sem = [None] * len(data)
    corr = [None] * len(data)
    indices = [None] * len(data)
    
    a=0
    for animal in data:
        movement_trace, spikes_trace, behavior_trace = get_traces_from_data(data, animal=animal, day=day, movement=movement, spikes=spikes, behavior=behavior, calcium=calcium, multisess = multisess)
        
        if indices_prev!= False:
            spikes_trace = spikes_trace[indices_prev[a]]
        
        spike_mean[a],spike_sem[a], lags[a], behavior_mean[a], behavior_sem[a], corr[a], indices[a] = time_locked_activity(movement_trace, spikes_trace, behavior_trace, frames, seltype, sig_seltype)
        a+=1
    return spike_mean, spike_sem, lags, behavior_mean, behavior_sem, corr, indices



def time_locked_activity(movement, spikes, behavior, frames, seltype = 'struggle',  sig_seltype = True):
    evoked_spike, evoked_behavior = get_evoked_responses(movement, spikes, behavior, frames_before=frames, frames_during=frames)
    
    if seltype == 'struggle' and sig_seltype == True:
        evoked_spike_struggle, indices = evoked_spike_struggle_selective(spikes, movement, evoked_spike)
    elif seltype == 'immobility' and sig_seltype == True:
        evoked_spike_struggle, indices = evoked_spike_immobility_selective(spikes, movement, evoked_spike)
    else:
        evoked_spike_struggle = evoked_spike
        indices = np.arange(len(evoked_spike))
    
    evoked_spikes = z_score_epochs(evoked_spike_struggle)
    spike_mean, spike_sem, spike_std = trial_averaged_spike(evoked_spikes)
    evoked_behavior =z_score_behavior(evoked_behavior)
    behavior_mean, behavior_sem, behavior_std = trial_averaged_behavior(evoked_behavior)
    
    spike_mean = standardize_transients2(spike_mean.T).T
    behavior_mean = standardize_transients2(behavior_mean.reshape(-1,1))[:,0]
    
    lags, corr = time_lag(spike_mean, behavior_mean, seltype)
    
    #if sig_seltype == True:
    spike_mean, lags, indices = sig_time_lag(spike_mean, corr, lags, seltype = seltype, indices=indices, sig_seltype = sig_seltype)
    
    return spike_mean,spike_sem, lags, behavior_mean, behavior_sem, corr, indices


#%% helper functions

def persistent_timing(indices1, indices3, indices9):
    d1_only = np.zeros(len(indices1))
    d1_and_3 = np.zeros(len(indices1))
    d1_and_9 = np.zeros(len(indices1))
    d1_and_3_and_9 = np.zeros(len(indices1))
    
    for i in range(len(indices1)):
        i1 = indices1[i]
        i3 = indices3[i]
        i9 = indices9[i]
    
        neurons = np.zeros((np.max(np.concatenate((i1,i3,i9)))+1,3))
        neurons[i1,0] = 1
        neurons[i3,1] = 1
        neurons[i9,2] = 1
        neurons = neurons.astype(bool)
        d1_only[i] = np.sum(neurons[:,0] & ~neurons[:,1] & ~neurons[:,2])
        d1_and_3[i] = np.sum(neurons[:,0] & neurons[:,1] & ~neurons[:,2])
        d1_and_9[i] = np.sum(neurons[:,0] & neurons[:,2] & ~neurons[:,1])
        d1_and_3_and_9[i] = np.sum(neurons[:,0] & neurons[:,2] & neurons[:,1])

    d1_only = np.sum(d1_only)
    d1_and_3 = np.sum(d1_and_3)
    d1_and_9 = np.sum(d1_and_9)
    d1_and_3_and_9 = np.sum(d1_and_3_and_9)
    
    ratios = np.array([d1_only, d1_and_3, d1_and_9, d1_and_3_and_9])
    
    return ratios


def fraction_shifts(lags):
    fractions = np.zeros(3)
    fractions[0] = np.sum(lags < 0)
    fractions[1] = np.sum(lags == 0)
    fractions[2] = np.sum(lags > 0)
    
    return fractions


def extract_sig_time_lags_2_days(lags1, lags3):
    lag1 = copy.deepcopy(lags1)
    lag3 = copy.deepcopy(lags3)
    for i in range(len(lag3)):
        index = np.where(lag3[i] != 999)
        lag1[i] = lag1[i][index]
        lag3[i] = lag3[i][index]
    return lag1, lag3

def get_traces_from_data(data, animal, day, movement, spikes, behavior, calcium=False, multisess = False):
        movement = copy.deepcopy(data[animal][day][movement])
        behavior = copy.deepcopy(data[animal][day][behavior])
        
        spikes = copy.deepcopy(data[animal][day][spikes])
        
        if multisess == True:
            spikes = spikes[data[animal][day]['assignments_filts']]
        
        if calcium==True:
            behavior = remove_edge_behavior(behavior)
            behavior = start_end(behavior)
        
        return movement, spikes, behavior


def time_lag(spikes, movement, seltype):
    spikes = copy.deepcopy(spikes)
    movement = copy.deepcopy(movement)
    lag = np.empty(len(spikes))
    sig2 = movement - np.mean(movement)
    corr = [None] * len(spikes)
    for neuron in range(len(spikes)):
        sig1 = spikes[neuron] - np.mean(spikes[neuron])
        correlogram = correlate(sig1, sig2, mode= 'same')
        corr[neuron] = correlogram/ (len(sig2) * np.std(sig1) * np.std(sig2)) # Normalization
        lags = correlation_lags(sig1.size, sig2.size, mode='same')
        
        if seltype == 'struggle':
            lag[neuron] = lags[np.argmax(correlogram)]
        elif seltype == 'immobility':
            lag[neuron] = lags[np.argmin(correlogram)]
       
        
    return lag, corr
        

def sig_time_lag(spikes, corr, lags, seltype, indices, sig_seltype):
    spikes = copy.deepcopy(spikes)
    corr = copy.deepcopy(corr)
    lags = copy.deepcopy(lags)
    
    if seltype == 'struggle':
        sig = np.max(corr, 1) > 0.657#0.76
    elif seltype == 'immobility':
        sig = np.min(corr,1) < -0.588# 0.49
    else:
        sig = [True]*len(spikes)
    
    if sig_seltype == True: 
        spikes = spikes[sig]
        lags = lags[sig]
        indices = indices[sig]
    else:
        exclude = ~sig
        spikes[exclude] = 999
        lags[exclude] = 999
        indices[exclude] = 999
        
    
    return spikes, lags, indices
    

def get_evoked_responses(movement, spikes, behavior, frames_before=6000, frames_during=6000):
        evoked_spike = np.empty((len(spikes),frames_before+frames_during, int(len(behavior)/2)))
        evoked_behavior = np.empty((int(len(behavior)/2), frames_before+frames_during))
                                
        i = 0
        for timepoint in behavior[::2]:
            if timepoint + frames_during < len(movement) and timepoint - frames_before > 0:
                evoked_spike[:,:,i] = spikes[:,  timepoint-frames_before:timepoint+frames_during]
                evoked_behavior[i] = movement[timepoint-frames_before:timepoint+frames_during]
                i+=1
        return evoked_spike, evoked_behavior
    
def trial_averaged_spike(evoked_spikes):
    spike_mean = np.mean(evoked_spikes,2)
    spike_sem = stats.sem(evoked_spikes,2, nan_policy= 'omit')
    spike_std = np.std(evoked_spikes,2)
    return spike_mean, spike_sem, spike_std

def trial_averaged_behavior(movement):
    movement_mean = np.mean(movement, 0)
    movement_sem = stats.sem(movement,0, nan_policy = 'omit')
    movement_std = np.std(movement,0)
    return movement_mean, movement_sem, movement_std


def z_score_behavior(movement):
    reference = int(len(movement[0])/5)
    for epoch in range(0, len(movement)):
        movement[epoch] = np.transpose(standardize_transients2(movement[epoch].reshape(-1,1)))
        mean = np.mean(movement[epoch, 0:reference])
        movement[epoch] = movement[epoch] - mean
    return movement

def z_score_epochs(evoked_spikes):
    evoked_spikes = copy.deepcopy(evoked_spikes)
    reference = int(len(evoked_spikes[0])/5)
    for epoch in range(0,len(evoked_spikes[0,0,:])):
        evoked_spikes[:,:,epoch] = np.transpose(standardize_transients2(evoked_spikes[:,:,epoch].T))
        mean = np.mean(evoked_spikes[:,0:reference,epoch],1)
        evoked_spikes[:,:,epoch] = evoked_spikes[:,:,epoch] - np.array(mean, ndmin=2).T # set reference to zero
    return evoked_spikes

def evoked_spike_struggle_selective(spikes, movement, evoked_spike):
    indices = np.arange(len(spikes))
    movement_cor = rate_coding.correlation(spikes,movement)
    evoked_spike_struggle = evoked_spike[movement_cor > 0.2]
    return evoked_spike_struggle, indices[movement_cor > 0.2]

def evoked_spike_immobility_selective(spikes, movement, evoked_spike):
    indices = np.arange(len(spikes))
    movement_cor = rate_coding.correlation(spikes,movement)
    evoked_spike_immobility = evoked_spike[movement_cor < -0.2]
    return evoked_spike_immobility, indices[movement_cor < -0.2]


def remove_edge_behavior(behavior):
    behavior_transform = copy.deepcopy(behavior)
    if behavior[0] == 1:
        e = 0
        while behavior[e] ==1:
            behavior_transform[e] = 0
            e+=1
    return behavior_transform 


def start_end(behavior):
    time=[]
    for i in range(1,len(behavior)-1):
        if behavior[i] == 1 and behavior[i-1] == 0:
            start_i = i
            time.append(start_i)
        if behavior[i] == 1 and behavior[i+1] == 0:
            end_i = i
            time.append(end_i)
    if behavior[-1] == 1:
        time.append(len(behavior))
    return time

def standardize_transients(transients):
    scaler = preprocessing.StandardScaler().fit(transients)
    transient_scaled = scaler.transform(transients)
    return transient_scaled

def standardize_transients2(transients):   
    scaler = preprocessing.MinMaxScaler().fit(transients)
    transient_scaled = scaler.transform(transients)
    return transient_scaled


def sort_map(data, sort, lags, subplot,fig, title):
    im = subplot.imshow(data[sort], interpolation='nearest', origin = 'lower', aspect='auto', cmap='jet')
    #subplot.set_yticklabels(lags)
    #subplot.set_xticks(range(1,20))
    subplot.axvline(x=len(data[0])/2, color ='white', lw=2)
    
    fig.colorbar(im)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.title.set_text(title)
    
def plot_pie(lags, subplot):
    labels = ['early', '', 'late']
    colors = ['grey', 'white', 'black']
    subplot.pie(lags, labels=labels, colors = colors, wedgeprops={"edgecolor":"k",'linewidth': 3,  'antialiased': True})


def plot_pie2(lags, subplot):
    labels = ['', 'd3', 'd9', 'd3 and 9']
    colors = ['white', 'red', 'blue','magenta']
    subplot.pie(lags, labels=labels, colors = colors, wedgeprops={"edgecolor":"k",'linewidth': 3,  'antialiased': True})
    