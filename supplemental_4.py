# -*- coding: utf-8 -*-


import functions.rate_coding
import numpy as np
import functions.Graphs
import matplotlib.pyplot as plt
import time_locked_functions
import functions.glm_to_predict_speed_from_neuronal_activity as glm
import functions.glm_tst_50_neurons_no_iterations as glm_50
import functions.statistics_test as st
#%% Load data
data = np.load('/datapath/dataset.npy', allow_pickle=True).item()

#%% A Predict struggle/immobility 
res, shuff =  rate_coding.decode_struggle_immobility(data, mod = 'SVC') 
fig, ax = plt.subplots(figsize= (3,5))
Graphs.boxplot2_with_points_and_lines(res[:,0], shuff[:,0], xlabel = ['data', 'shuff'], title = 'SVC', subplot=ax, ylim=[0.3,1])
stats_svc = st.post_hoc_paired_multiple([res[:,0]], [shuff[:,0]])

#%% B GaussianNB
res, shuff =  rate_coding.decode_struggle_immobility(data, mod = 'GaussianNB') 
fig, ax = plt.subplots(figsize= (3,5))
Graphs.boxplot2_with_points_and_lines(res[:,0], shuff[:,0], xlabel = ['data', 'shuff'], title = 'GaussianNB', subplot=ax, ylim=[0.3,1])
stats_gaus = st.post_hoc_paired_multiple([res[:,0]], [shuff[:,0]])

#%% C RidgeClassifier
res, shuff =  rate_coding.decode_struggle_immobility(data, mod = 'RidgeClassifier') 
fig, ax = plt.subplots(figsize= (3,5))
Graphs.boxplot2_with_points_and_lines(res[:,0], shuff[:,0], xlabel = ['data', 'shuff'], title = 'RidgeClassifier', subplot=ax, ylim=[0.3,1])
stats_ridge = st.post_hoc_paired_multiple([res[:,0]], [shuff[:,0]])

#%% D E F
glm_pred_speed = glm.fig2()
glm_50_neurons = glm_50.supplemental4()

#%% G
frames =10
spike_mean,spike_sem, lags, behavior_mean, behavior_sem, corr, indices = time_locked_functions.Time_locked_activity(data = data, frames = frames, day = 'day1', movement ='tst_speed_dlc', 
                spikes='fn_tst', behavior='immobility_time', seltype = 'immobility', calcium=True)


spike_mean_all = np.vstack(spike_mean)
lags_all = np.hstack(lags)

sort_lags = np.argsort(lags_all,0)
lags_sorted = lags_all[sort_lags]
fig, ax = plt.subplots(figsize=(5, 10))
time_locked_functions.sort_map(spike_mean_all, sort_lags, lags_sorted, ax, fig, title ='')

lags_pie = time_locked_functions.fraction_shifts(lags_all)
fig, ax = plt.subplots()
time_locked_functions.plot_pie(lags_pie, ax)

#%% H
frames =10
spike_mean,spike_sem, lags, behavior_mean, behavior_sem, corr, indices = time_locked_functions.Time_locked_activity(data = data, frames = frames, day = 'day1', movement ='tst_speed_dlc', 
                spikes='fn_tst', behavior='struggle_time', seltype = 'struggle', calcium=True)


spike_mean_all = np.vstack(spike_mean)
lags_all = np.hstack(lags)


sort_lags = np.argsort(lags_all,0)
lags_sorted = lags_all[sort_lags]
fig, ax = plt.subplots(figsize=(5, 10))
time_locked_functions.sort_map(spike_mean_all, sort_lags, lags_sorted, ax, fig, title ='')

lags_pie = time_locked_functions.fraction_shifts(lags_all)
fig, ax = plt.subplots()
time_locked_functions.plot_pie(lags_pie, ax)


#%% I Compare speed differences for all bodyparts
# baseline_speed_bodyparts, tst_speed_bodyparts
label =  ['nose', 'neck', 'leftbody', 'rightbody', 'basetail']
speed_bodyparts =  rate_coding.get_mean_speed_bodyparts(data, 'day1', 'baseline_speed_bodyparts')
corr_mat = np.mean(rate_coding.correlation_matrix_bodyparts(data, 'day1', 'baseline_speed_bodyparts'),0)
Graphs.corr_matrix(corr_mat, label)
fig, ax = plt.subplots(figsize= (5,3))
Graphs.boxplot5_with_points_and_lines(speed_bodyparts[:,0],speed_bodyparts[:,1],speed_bodyparts[:,2],speed_bodyparts[:,3],speed_bodyparts[:,4], xlabel =label, subplot=ax, title='')

#%%
label = ['camera', 'neck', 'leftfoot', 'rightfoot']
speed_bodyparts =  rate_coding.get_mean_speed_bodyparts(data, 'day1', 'tst_speed_bodyparts')
corr_mat = np.mean(rate_coding.correlation_matrix_bodyparts(data, 'day1', 'tst_speed_bodyparts'),0)
Graphs.corr_matrix(corr_mat, label)
fig, ax = plt.subplots(figsize= (5,3))
Graphs.boxplot4_with_points_and_lines(speed_bodyparts[:,0],speed_bodyparts[:,1],speed_bodyparts[:,2],speed_bodyparts[:,3], xlabel =label, subplot=ax, title='')

