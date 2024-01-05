# -*- coding: utf-8 -*-

import functions.rate_coding
import numpy as np
import functions.Graphs
import matplotlib.pyplot as plt
import functions.statistics_test as st
import functions.glm_to_predict_speed_from_neuronal_activity as glm
#%% Load data
data = np.load('/datapath/dataset.npy', allow_pickle=True).item()


#%% 2A Predict struggle/immobility 
res, shuff =  rate_coding.decode_struggle_immobility(data) 
fig, ax = plt.subplots(figsize=[3,5])
Graphs.boxplot2_with_points_and_lines(res[:,0],shuff[:,0], ylim=[0.3,1], xlabel=['data', 'shuffle'], title ='', subplot=ax)
stats_res = st.post_hoc_paired_multiple([res[:,0]], [shuff[:,0]])
#%%  2B correlate movement selectivity between struggle and immobility 
tst_movement_tuning = rate_coding.movement_tuning(data = data, transientname= 'fn_tst', day = 'day1', behaviorname = 'tst_speed_dlc')
baseline_movement_tuning = rate_coding.movement_tuning(data = data, transientname = 'fn_baseline', day = 'day1', behaviorname = 'baseline_speed_dlc')

mixlm_snr = st.mixedlm(tst_movement_tuning, baseline_movement_tuning)
print(mixlm_snr) 
fig, ax = plt.subplots(figsize=(5, 5))
Graphs.correlation_plot(np.hstack(baseline_movement_tuning), np.hstack(tst_movement_tuning), subplot = ax)

ab_tst_movement_tuning = [np.mean(np.absolute(tst_movement_tuning[a])) for a in range(len(tst_movement_tuning))]
ab_baseline_movement_tuning = [np.mean(np.absolute(baseline_movement_tuning[a])) for a in range(len(baseline_movement_tuning))]
fig, ax = plt.subplots(igsize=[3,5])
Graphs.boxplot2_with_points_and_lines(ab_baseline_movement_tuning, ab_tst_movement_tuning, xlabel = ['baseline', 'TS'], title = '', subplot=ax, ylim=[0,0.2])

#%% 2C  and D
pred_speed = glm.fig2()

#%% 2F Predict from tst
pred_between_ts = rate_coding.decode_movement_between_tst_baseline(data, 'day1', pred_from= 'tst')
pred_between_bs = rate_coding.decode_movement_between_tst_baseline(data, 'day1', pred_from= 'baseline')

fig, ax = plt.subplots(figsize=[3,5])
Graphs.boxplot3_with_points_and_lines(pred_between_ts['between_res'],pred_between_ts['within_res'], pred_between_ts['shuff'], subplot = ax, xlabel = ['baseline', 'tst', 'shuffle'], title = 'predict movement from tst', ylim=[-0.25,1])




fig, ax = plt.subplots(figsize=[3,5])
Graphs.boxplot3_with_points_and_lines(pred_between_bs['within_res'], pred_between_bs['between_res'], pred_between_bs['shuff'], subplot = ax, xlabel = ['baseline', 'tst', 'shuffle'], title = 'predict movement from tst', ylim=[-0.3,1])

stats_pred = st.post_hoc_paired_multiple([pred_between_ts['within_res'], pred_between_ts['within_res'], pred_between_ts['between_res'],    pred_between_bs['within_res'], pred_between_bs['within_res'], pred_between_bs['between_res'], pred_between_ts['within_res'] ] ,
                                         
                                         [pred_between_ts['between_res'], pred_between_ts['shuff'], pred_between_ts['shuff'],     pred_between_bs['between_res'], pred_between_bs['shuff'], pred_between_bs['shuff'], pred_between_bs['within_res'] ]
                                         )

# trained on ts:
    # ts vs baseline
    # ts vs shuffle
    # baseline vs shuffle
    
# trained on baseline
    # baseline vs ts
    # baseline vs shuffle
    # ts vs shuffle

# trained both
    # trained ts predicing ts vs trained baseline predicting baseline

