# -*- coding: utf-8 -*-

#%%
import functions.rate_coding
import numpy as np
import functions.Graphs
import matplotlib.pyplot as plt
import pandas as pd
import functions.statistics_test as st
#%% Load data
data = np.load('/datapath/dataset.npy', allow_pickle=True).item()


#%% 1C Behavioral performance
performance = rate_coding.behavior_performance(data)

fig, ax = plt.subplots(1,3)
Graphs.boxplot3_with_points_and_lines(performance['immobility_time'][:,0], performance['immobility_time'][:,1], performance['immobility_time'][:,2], xlabel='', title ='', subplot=ax[0], ylim=[0,300])
Graphs.boxplot3_with_points_and_lines(performance['struggle_time'][:,0], performance['struggle_time'][:,1], performance['struggle_time'][:,2], xlabel='', title ='', subplot=ax[1], ylim=[0,300])
Graphs.boxplot3_with_points_and_lines(performance['movement_mean'][:,0], performance['movement_mean'][:,1], performance['movement_mean'][:,2], xlabel='', title ='', subplot=ax[2], ylim=[0,50])

stats_immobility = st.post_hoc_paired_multiple([performance['immobility_time'][:,0], performance['immobility_time'][:,0], performance['immobility_time'][:,1]] , [performance['immobility_time'][:,1], performance['immobility_time'][:,2], performance['immobility_time'][:,2]])
stats_struggle = st.post_hoc_paired_multiple([performance['struggle_time'][:,0], performance['struggle_time'][:,0], performance['struggle_time'][:,1]] , [performance['struggle_time'][:,1], performance['struggle_time'][:,2], performance['struggle_time'][:,2]])
stats_movement = st.post_hoc_paired_multiple([performance['movement_mean'][:,0], performance['movement_mean'][:,0], performance['movement_mean'][:,1]] , [performance['movement_mean'][:,1], performance['movement_mean'][:,2], performance['movement_mean'][:,2]])

#%% 1D Nr active neurons
active_neurons = rate_coding.active_neurons(data)
fig, ax = plt.subplots()
Graphs.boxplot3_with_points_and_lines(active_neurons['d1_active'], active_neurons['d3_active'], active_neurons['d9_active'], xlabel='', title ='detected neurons', subplot=ax, ylim=[0,340])
#%% 1D SNR
snr1, snr3, snr9 = rate_coding.snr(data)

# remove inf values
for d in range(len(snr1)):
    snr9[d] = snr9[d][snr1[d]!=np.inf]
    snr1[d] = snr1[d][snr1[d]!=np.inf]

    snr1[d] = snr1[d][snr9[d]!=np.inf]
    snr9[d] = snr9[d][snr9[d]!=np.inf]
    
mixlm_snr = st.mixedlm(snr1, snr9)
print(mixlm_snr)

#%% 1E Selectivity index tst v baseline
selec = rate_coding.tst_v_baseline_selectivity(data, day = 'day1', transient_rate=False)
selec_all = np.hstack(selec)
selec_all_pandas = pd.DataFrame({'data':selec_all, 'x':np.ones(len(selec_all))}) 



Graphs.violin(selec_all_pandas)

ts_selective = np.zeros(len(data))
baseline_selective = np.zeros(len(data))

for a in range(len(data)):
    selec_a = np.asarray(selec[a])
    ts_selective[a] = len(selec_a[selec_a>=0.2])/len(selec_a)
    baseline_selective[a] = len((selec_a[selec_a<-0.2]))/len(selec_a)

fig, ax = plt.subplots()
Graphs.boxplot2_with_points_and_lines(baseline_selective, ts_selective, xlabel = ['baseline', 'TS'], title = '', subplot=ax, ylim = [0,0.5])


#%% 1F Correlate selectivity index over days
selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day9')
fig, ax = plt.subplots(figsize=(4, 5))
Graphs.correlation_plot(np.hstack(selec1), np.hstack(selec2), subplot = ax, xlim = [-1,1], ylim=[-1,1])
mixlm_snr = st.mixedlm(selec1, selec2)
print(mixlm_snr) 


#%% 1G Correlate selectivity index over days per mouse
selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day3')
d1_d3 = rate_coding.selectivity_corr(selec1, selec2)

selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day9')
d1_d9 = rate_coding.selectivity_corr(selec1, selec2)

selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day3', day2 = 'day9')
d3_d9 = rate_coding.selectivity_corr(selec1, selec2)

res,res_total = st.one_way_repeated_measures_anova_general_three_groups(d1_d3,d3_d9, d1_d9)
t = st.post_hoc_paired_multiple([d1_d3, d1_d3, d3_d9], [d3_d9, d1_d9, d1_d9])

fig, ax = plt.subplots()
Graphs.boxplot3_with_points_and_lines(d1_d3, d3_d9,d1_d9,  xlabel='', title ='correlation', subplot=ax, ylim=[0,1])

#%% 1G Correlate selectivity scores across days with only selective neurons
selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day3')
sel_corr_3 = rate_coding.selectivity_corr_selectivity_neurons(selec1, selec2)

selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day9')
sel_corr_9 = rate_coding.selectivity_corr_selectivity_neurons(selec1, selec2)

selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day3', day2 = 'day9')
sel_corr_3_9 = rate_coding.selectivity_corr_selectivity_neurons(selec1, selec2)

res,res_total = st.one_way_repeated_measures_anova_general_three_groups(sel_corr_3,sel_corr_3_9, sel_corr_9)

fig, ax = plt.subplots(1, figsize=[3,5])
Graphs.boxplot3_with_points_and_lines(sel_corr_3,sel_corr_3_9, sel_corr_9,  xlabel='', title ='correlation', subplot=ax, ylim=[0,1])

#%% 1H
d1, d3, d1_shuff1, d3_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day3', 'day1')
_, d9, _, d9_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day9', 'day1')
shuff = np.mean((d3_shuff2, d9_shuff2), 0)

fig, ax = plt.subplots(1, figsize=[3,5])
Graphs.boxplot4_with_points_and_lines(d1,d3,d9,shuff, xlabel = ['d1', 'd3', 'd9', 'shuffle'], title = '', subplot = ax, ylim=[0.4,1.1])
t = st.post_hoc_paired_multiple([d1,d1,d1,d3,d9], [d3,d9,d1_shuff1,d3_shuff2,d9_shuff2])

#%%  1HDecode tst/baseline across days  only selective neurons
d1, d3, d1_shuff1, d3_shuff2 = rate_coding.decode_tst_baseline_days_selective_neurons(data, 'day3', 'day1')
_, d9, _, d9_shuff2 = rate_coding.decode_tst_baseline_days_selective_neurons(data, 'day9', 'day1')
shuff = np.mean((d3_shuff2, d9_shuff2), 0)
t = st.post_hoc_paired_multiple([d1,d1,d1,d3,d9], [d3,d9,shuff,shuff,shuff])

fig, ax = plt.subplots(1, figsize=[3,5])
Graphs.boxplot4_with_points_and_lines(d1,d3,d9,shuff, xlabel = ['d1', 'd3', 'd9', 'shuffle'], title = '', subplot = ax, ylim=[0.4,1.1])






