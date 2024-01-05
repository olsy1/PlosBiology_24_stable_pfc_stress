# -*- coding: utf-8 -*-


import functions.rate_coding
import numpy as np
import functions.Graphs
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import functions.statistics_test as st
#%% Load data

data = np.load('/datapath/dataset.npy', allow_pickle=True).item()

#%% 3A Brain region specific data
prl = [1, 1, 0.73, 0.64, 0.55, 0.67]
cg = [0, 0, 0.27, 0.36, 0.3, 0.33]
m2 = [0, 0, 0, 0, 0.15, 0]
results = rate_coding.neurons_per_regions(data, prl, cg, m2, 'day1', multisess=False)

#%% 3B
results1 = rate_coding.neurons_per_regions(data, prl, cg, m2, 'day1', multisess=True)
results3 = rate_coding.neurons_per_regions(data, prl, cg, m2, 'day3', multisess=True)
results9 = rate_coding.neurons_per_regions(data, prl, cg, m2, 'day9', multisess=True)

#%%
d1 = np.hstack(results1['ts_selectivity_pr'])
d1 = np.array(d1[d1 != np.array(None)], dtype= 'float64') 
d3 = np.hstack(results9['ts_selectivity_pr'])
d3 = np.array(d3[d3 != np.array(None)], dtype= 'float64') 

fig, ax = plt.subplots(figsize=(5, 5))
Graphs.correlation_plot(d1, d3, subplot = ax)

stats.pearsonr(d1,d3)
#%%
# pandas

baseline_mov = np.hstack(results['baseline_movement_tuning_cg'])
ts_mov = np.hstack(results['ts_movement_tuning_cg'])
ts_mov = ts_mov[ts_mov != np.array(None)]                  
baseline_mov = baseline_mov[baseline_mov != np.array(None)]                         

fig, ax = plt.subplots(figsize=(5, 5))
Graphs.correlation_plot(baseline_mov, ts_mov, subplot = ax)

stats.pearsonr(baseline_mov, ts_mov)


#%%
selec_all = np.hstack(results['ts_selectivity_m2'])
selec_all = np.array(selec_all[selec_all != np.array(None)], dtype= 'float64') 

selec_all_pandas = pd.DataFrame({'data':selec_all, 'x':np.ones(len(selec_all))}) 
Graphs.violin(selec_all_pandas)

#%%
selec_all = np.hstack(results['ts_selectivity_cg'])
selec_cg = np.array(selec_all[selec_all != np.array(None)], dtype= 'float64') 

selec_all = np.hstack(results['ts_selectivity_pr'])
selec_pr = np.array(selec_all[selec_all != np.array(None)], dtype= 'float64') 

stats.ttest_ind(selec_cg, selec_pr)

 
#%% 3C Selectivity indices
males = [0,3,4,5]
females = [1,2]

selec = rate_coding.tst_v_baseline_selectivity(data, day = 'day1', transient_rate=False)
selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day3')

males_selec = [selec[i] for i in males]
females_selec = [selec[i] for i in females]
stats.ttest_ind(np.hstack(males_selec), np.hstack(females_selec))

selec_all = np.hstack(females_selec)

selec_all_pandas = pd.DataFrame({'data':selec_all, 'x':np.ones(len(selec_all))}) 
ts_selective = len((selec_all[selec_all>0.5]))/len(selec_all)*100
baseline_selective = len((selec_all[selec_all<-0.5]))/len(selec_all)*100
Graphs.violin(selec_all_pandas)



#%%
corr = np.empty(len(data))
for i in range(len(corr)):
    corr[i] = stats.pearsonr(selec1[i],selec2[i])[0]
males_corr = np.array([corr[i] for i in males])
females_corr = np.array([corr[i] for i in females])
stats.ttest_ind(males_corr, females_corr)

Graphs.boxplot2_with_points(males_corr, females_corr)

#%% 3D
sex = females
selec1, selec2 = rate_coding.tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day9')
selec_s1 = [selec1[i] for i in sex]
selec_s2 = [selec2[i] for i in sex]

stats.pearsonr(np.hstack(selec_s1), np.hstack(selec_s2))

fig, ax = plt.subplots(figsize=(4, 5))
Graphs.correlation_plot(np.hstack(selec_s1), np.hstack(selec_s2), subplot = ax, xlim = [-1,1], ylim=[-1,1])

mixlm_snr = st.mixedlm(selec_s1, selec_s2)
print(mixlm_snr)


#%% 3E Support vector machine
d1, d3, d1_shuff1, d3_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day3', 'day1', model_type ='SVC')
_, d9, _, d9_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day9', 'day1', model_type = 'SVC')
shuff = np.mean((d3_shuff2, d9_shuff2), 0)

fig, ax = plt.subplots()
Graphs.boxplot4_with_points_and_lines(d1,d3,d9,shuff, xlabel = ['d1', 'd3', 'd9', 'shuffle'], title = 'Support vector machine', subplot = ax, ylim=[0.4,1.1])

#%% GaussianNB
d1, d3, d1_shuff1, d3_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day3', 'day1', model_type ='GaussianNB')
_, d9, _, d9_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day9', 'day1', model_type = 'GaussianNB')
shuff = np.mean((d3_shuff2, d9_shuff2), 0)

fig, ax = plt.subplots()
Graphs.boxplot4_with_points_and_lines(d1,d3,d9,shuff, xlabel = ['d1', 'd3', 'd9', 'shuffle'], title = 'GaussianNB', subplot = ax, ylim=[0.4,1])

#%% RidgeClassifier
d1, d3, d1_shuff1, d3_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day3', 'day1', model_type ='RidgeClassifier')
_, d9, _, d9_shuff2 = rate_coding.decode_tst_baseline_days(data, 'day9', 'day1', model_type = 'RidgeClassifier')
shuff = np.mean((d3_shuff2, d9_shuff2), 0)

fig, ax = plt.subplots()
Graphs.boxplot4_with_points_and_lines(d1,d3,d9,shuff, xlabel = ['d1', 'd3', 'd9', 'shuffle'], title = 'RidgeClassifier', subplot = ax, ylim=[0.4,1])

