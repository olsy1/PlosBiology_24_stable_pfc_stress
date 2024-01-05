# -*- coding: utf-8 -*-


import functions.rate_coding
import numpy as np
import functions.Graphs
import matplotlib.pyplot as plt
from scipy import stats
import functions.statistics_test as st
#%% Load data
data = np.load('/datapath/dataset.npy', allow_pickle=True).item()


#%% A Prop struggle/immo cells over days
selectivity1 = rate_coding.movement_tuning(data, 'day1', 'fn_tst', 'tst_speed_dlc')
selectivity3 = rate_coding.movement_tuning(data, 'day3', 'fn_tst', 'tst_speed_dlc')
selectivity9 = rate_coding.movement_tuning(data, 'day9', 'fn_tst', 'tst_speed_dlc')

struggle_ratio1, immobility_ratio1 = rate_coding.selectivity_proportions(selectivity1)
struggle_ratio3, immobility_ratio3 = rate_coding.selectivity_proportions(selectivity3)
struggle_ratio9, immobility_ratio9 = rate_coding.selectivity_proportions(selectivity9)

fig, ax = plt.subplots()
Graphs.boxplot3_with_points_and_lines(struggle_ratio1,struggle_ratio3,struggle_ratio9,['d1', 'd3', 'd9'], title = 'proportion struggle selective neurons', subplot=ax, ylim=[0,1])
res,res_total = st.one_way_repeated_measures_anova_general_three_groups(struggle_ratio1, struggle_ratio3, struggle_ratio9)
a_struggle_stats = st.post_hoc_paired_multiple([struggle_ratio1, struggle_ratio1, struggle_ratio3] , [struggle_ratio3, struggle_ratio9, struggle_ratio9])


fig, ax = plt.subplots()
Graphs.boxplot3_with_points_and_lines(immobility_ratio1, immobility_ratio3, immobility_ratio9,['d1', 'd3', 'd9'], title = 'proportion immobility selective neurons', subplot=ax, ylim = [0,1])
res,res_total = st.one_way_repeated_measures_anova_general_three_groups(immobility_ratio1, immobility_ratio3, immobility_ratio9)
a_immobility_stats = st.post_hoc_paired_multiple([immobility_ratio1, immobility_ratio1, immobility_ratio3] , [immobility_ratio3, immobility_ratio9, immobility_ratio9])


#%% B area under the curve during struggle over days
activity_s1 = rate_coding.activity_during_behavior(data, 'day1', 'fn_tst', 'struggle_time', transient_rate=False)
activity_s3 = rate_coding.activity_during_behavior(data, 'day3', 'fn_tst', 'struggle_time', transient_rate=False)
activity_s9 = rate_coding.activity_during_behavior(data, 'day9', 'fn_tst', 'struggle_time', transient_rate=False)

fig, ax = plt.subplots()
Graphs.boxplot3_with_points_and_lines(activity_s1,activity_s3,activity_s9,['d1', 'd3', 'd9'], title = 'mouse averaged mean z-transient rate during struggle for struggle cells', subplot=ax, ylim = [0,1])
res,res_total = st.one_way_repeated_measures_anova_general_three_groups(activity_s1, activity_s3, activity_s9)
b_struggle_stats = st.post_hoc_paired_multiple([activity_s1, activity_s1, activity_s3] , [activity_s3, activity_s9, activity_s9])

#%% B area under the curve during immobility over days
activity_s1 = rate_coding.activity_during_behavior(data, 'day1', 'fn_tst', 'immobility_time', transient_rate=False)
activity_s3 = rate_coding.activity_during_behavior(data, 'day3', 'fn_tst', 'immobility_time', transient_rate=False)
activity_s9 = rate_coding.activity_during_behavior(data, 'day9', 'fn_tst', 'immobility_time', transient_rate=False)

fig, ax = plt.subplots()
Graphs.boxplot3_with_points_and_lines(activity_s1,activity_s3,activity_s9,['d1', 'd3', 'd9'], title = 'mouse averaged mean z-calcium activity during immobility for immobility cells', subplot=ax, ylim = [0,1])
res,res_total = st.one_way_repeated_measures_anova_general_three_groups(activity_s1, activity_s3, activity_s9)
b_immobility_stats = st.post_hoc_paired_multiple([activity_s1, activity_s1, activity_s3] , [activity_s3, activity_s9, activity_s9])


#%% C Stability within days. Are neurons movementselective within a day? Compare first and second half
mov_tun1, mov_tun2 = rate_coding.movement_tuning_first_v_second(data = data, transientname= 'fn_tst', day = 'day1', behaviorname = 'tst_speed_dlc')

fig, ax = plt.subplots(figsize=(5, 5))
Graphs.correlation_plot(np.hstack(mov_tun1), np.hstack(mov_tun2), subplot = ax, xlim = [-1,1], ylim=[-1,1])

mixlm_snr = st.mixedlm(mov_tun1, mov_tun2)
print(mixlm_snr) 


#%% D Compare males and females correlatate movement selectivity across days
males = [0,3,4,5]
females = [1,2]

sex = males

selec1, selec2, _ = rate_coding.movement_tuning_across_days(data, 'day1', 'day3', 'fn_tst', 'tst_speed_dlc')

selec_s1 = [selec1[i] for i in sex]
selec_s2 = [selec2[i] for i in sex]

stats.pearsonr(np.hstack(selec_s1), np.hstack(selec_s2))

fig, ax = plt.subplots(figsize=(5, 5))
Graphs.correlation_plot(np.hstack(selec_s1), np.hstack(selec_s2), subplot = ax, xlim = [-0.6,0.8], ylim=[-0.6,0.8])
mixlm_snr = st.mixedlm(selec_s1, selec_s2)
print(mixlm_snr)



tst_movement_tuning = rate_coding.movement_tuning(data = data, transientname= 'fn_tst', day = 'day1', behaviorname = 'tst_speed_dlc')
males = [0,3,4,5]
females = [1,2]

male_res = [res[i,0] for i in males]
female_res = [res[i,0] for i in females]
stats.ttest_ind(male_res, female_res)


male_tuning = [tst_movement_tuning[i] for i in males]
female_tuning = [tst_movement_tuning[i] for i in females]
stats.ttest_ind(np.hstack(male_tuning), np.hstack(female_tuning))





