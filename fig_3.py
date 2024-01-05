# -*- coding: utf-8 -*-

import functions.rate_coding
import numpy as np
import functions.manifold
import functions.Graphs
import matplotlib.pyplot as plt
import functions.statistics_test as st
from scipy import stats
#%% Load data
data = np.load('/datapath/dataset.npy', allow_pickle=True).item()


#%% 3B correlate movement selectivity across days 
tuning1_d3, tuning2_d3, stability_d3 = rate_coding.movement_tuning_across_days(data, 'day1', 'day3', 'fn_tst', 'tst_speed_dlc')
tuning1_d9, tuning2_d9, stability_d9 = rate_coding.movement_tuning_across_days(data, 'day1', 'day9', 'fn_tst', 'tst_speed_dlc')

fig, ax = plt.subplots(figsize=(5, 5))
Graphs.correlation_plot(np.hstack(tuning1_d3), np.hstack(tuning2_d3), subplot = ax, xlim = [-0.6,0.8], ylim=[-0.6,0.8])

mixlm_snr = st.mixedlm(tuning1_d3, tuning2_d3)
print(mixlm_snr) 
t, p = stats.ttest_rel(stability_d3, stability_d9)

fig, ax = plt.subplots()
Graphs.boxplot2_with_points_and_lines(stability_d3, stability_d9, xlabel = ['d3', 'd9'], title = '', subplot=ax, ylim=[0,0.7])


#%% 3C Decode struggle/immobility across days
behavior_d3, behavior_d1, shuffl_3 = rate_coding.decode_struggle_immobility_between_days(data, 'day3', 'day1')
behavior_d9, behavior_d1, shuffl_9 = rate_coding.decode_struggle_immobility_between_days(data, 'day9', 'day1')

stats = st.post_hoc_paired_multiple([behavior_d1, shuffl_3, behavior_d1, shuffl_9] , [behavior_d3, behavior_d3, behavior_d9, behavior_d9])


fig, ax = plt.subplots(1,figsize=[4,5])
Graphs.boxplot4_with_points_and_lines(behavior_d1,behavior_d3, behavior_d9,shuffl_3,   subplot = ax, xlabel = ['','', '--', '-'], title = 'Decode struggle/immobility', ylim=[0.3,1])


#%% 3C decode speed across days
behavior_d3, behavior_d1, shuffl_3 = rate_coding.decode_speed_between_days(data, 'day3', 'day1') 
behavior_d9, behavior_d1, shuffl_9 = rate_coding.decode_speed_between_days(data, 'day9', 'day1')

fig, ax = plt.subplots(1,figsize=[4,5])
Graphs.boxplot4_with_points_and_lines(behavior_d1,behavior_d3, behavior_d9,shuffl_3,   subplot = ax, xlabel = ['','', '--', '-'], title = 'Decode struggle/immobility', ylim=[-0.2, 1])
stats = st.post_hoc_paired_multiple([behavior_d1, shuffl_3, behavior_d1, shuffl_3] , [behavior_d3, behavior_d3, behavior_d9, behavior_d9])


#%% Embed manifolds for each day
res_day1, order_day1, shuff1 = manifold.manifolds(data, 'day1', 'struggle_time', 'immobility_time') 
res_day3, order_day3, shuff3 = manifold.manifolds(data, 'day3', 'struggle_time', 'immobility_time') 
res_day9, order_day9, shuff9 = manifold.manifolds(data, 'day9', 'struggle_time', 'immobility_time') 


#%% 3F align manifolds across days
transients_mean_day1, order_mean_day1 = manifold.manifolds_mean(data,'day1', 'struggle_time', 'immobility_time', res_day1) # mean of manifolds
transients_mean_day3, order_mean_day3 = manifold.manifolds_mean(data,'day3', 'struggle_time', 'immobility_time', res_day3)
transients_mean_day9, order_mean_day9 = manifold.manifolds_mean(data,'day9', 'struggle_time', 'immobility_time', res_day9)

res_align_3, b, error, t_3 = manifold.align_manifold_days(transients_mean_day3, transients_mean_day1) # align all manifold
res_align_9, b, error, t_9 = manifold.align_manifold_days(transients_mean_day9, transients_mean_day1) # align all manifold

res_reference = manifold.scale_translate(res_day1)# translate and scale reference matrix
res_transform_3, res_non_transform_3, Res_many_non_transform_3 = manifold.rotation(res_day3, t_3, average = True, rots = 100) # apply rotation to full data
res_transform_9, res_non_transform_9, Res_many_non_transform_9 = manifold.rotation(res_day9, t_9, average = True, rots = 100) 

################################## shuffle #################################################################
transients_mean_day3_shuff, order_mean_day3_shuff = manifold.manifolds_mean(data,'day3', 'struggle_time', 'immobility_time', shuff3)
transients_mean_day9_shuff, order_mean_day9_shuff = manifold.manifolds_mean(data,'day9', 'struggle_time', 'immobility_time', shuff9)

res_align_3_shuff, b_shuff, error_shuff, t_3_shuff = manifold.align_manifold_days(transients_mean_day3_shuff, transients_mean_day1) # align all manifold
res_align_9_shuff, b_shuff, error_shuff, t_9_shuff = manifold.align_manifold_days(transients_mean_day9_shuff, transients_mean_day1) # align all manifold

res_transform_3_shuff, res_non_transform_3_shuff, Res_many_non_transform_3_shuff = manifold.rotation(shuff3, t_3_shuff, average = True, rots =100) # a
res_transform_9_shuff, res_non_transform_9_shuff, Res_many_non_transform_9_shuff  = manifold.rotation(shuff9, t_9_shuff, average = True, rots=100) 


#%% 3F Predict struggle/immobility between days from the manifolds
res_manifold_aligned2_3, res_manifold_aligned1_3 = rate_coding.decode_struggle_immobility_between_manifolds_days2(data, 'day3', 'day1', res_transform_3, res_reference)
res_manifold_non_aligned2_3 = rate_coding.decode_struggle_immobility_between_manifolds_days2_rotations(data, 'day3', 'day1', Res_many_non_transform_3,res_reference)
res_manifold_aligned2_shuff_3, res_manifold_aligned1_shuff_3 = rate_coding.decode_struggle_immobility_between_manifolds_days2(data, 'day3', 'day1', res_transform_3_shuff, res_reference)

res_manifold_aligned2_9, res_manifold_aligned1_9 = rate_coding.decode_struggle_immobility_between_manifolds_days2(data, 'day9', 'day1', res_transform_9, res_reference)
res_manifold_non_aligned2_9  = rate_coding.decode_struggle_immobility_between_manifolds_days2_rotations(data, 'day9', 'day1', Res_many_non_transform_9,res_reference)
res_manifold_aligned2_shuff_9, res_manifold_aligned1_shuff_9 = rate_coding.decode_struggle_immobility_between_manifolds_days2(data, 'day9', 'day1', res_transform_9_shuff, res_reference)

fig, ax = plt.subplots(1,figsize=[4,5])
Graphs.boxplot4_with_points_and_lines(res_manifold_aligned1_3, res_manifold_aligned2_3, res_manifold_non_aligned2_3, res_manifold_aligned2_shuff_3,  subplot = ax, xlabel = ['day1','day3-rotated', 'day3-random-rotated', 'day3-shuffle'], title = 'Decode struggle/immobility', ylim=[0.4,1])

res, res_total = st.one_way_repeated_measures_anova_general_four_groups(res_manifold_aligned1_3, res_manifold_aligned2_3, res_manifold_non_aligned2_3, res_manifold_aligned2_shuff_3)
stats_res = st.post_hoc_paired_multiple([res_manifold_aligned2_3, res_manifold_aligned2_3, res_manifold_aligned2_3] , [res_manifold_aligned1_3, res_manifold_non_aligned2_3, res_manifold_aligned2_shuff_3])

fig, ax = plt.subplots(1,figsize=[3,5])
Graphs.boxplot3_with_points_and_lines(res_manifold_aligned2_9, res_manifold_non_aligned2_9, res_manifold_aligned2_shuff_9,  subplot = ax, xlabel = ['day9-rotated', 'day9-random-rotated', 'day9-shuffle'], title = 'Decode struggle/immobility',  ylim=[0.4,1])
stats_res = st.post_hoc_paired_multiple([res_manifold_aligned1_3, res_manifold_aligned1_3, res_manifold_aligned1_3] , [res_manifold_aligned2_9, res_manifold_non_aligned2_9, res_manifold_aligned2_shuff_9])

#%% 3F Predict speed from manifolds between days 
res_aligned_3 = rate_coding.decode_movement_between_manifold_days2(data, 'day3', 'day1', res_transform_3, res_reference)
res_non_aligned_3  = rate_coding.decode_movement_between_manifold_days_rotations2(data, 'day3', 'day1', Res_many_non_transform_3, res_reference)

res_aligned_shuff_3 = rate_coding.decode_movement_between_manifold_days2(data, 'day3', 'day1', res_transform_3_shuff, res_reference)
res_non_aligned_shuff_3 = rate_coding.decode_movement_between_manifold_days2(data, 'day3', 'day1', res_non_transform_3_shuff, res_reference)

res_aligned_9 = rate_coding.decode_movement_between_manifold_days2(data, 'day9', 'day1', res_transform_9, res_reference)
res_non_aligned_9  = rate_coding.decode_movement_between_manifold_days_rotations2(data, 'day9', 'day1', Res_many_non_transform_9, res_reference)

res_aligned_shuff_9 = rate_coding.decode_movement_between_manifold_days2(data, 'day9', 'day1', res_transform_9_shuff, res_reference)
res_non_aligned_shuff_9 = rate_coding.decode_movement_between_manifold_days2(data, 'day9', 'day1', res_non_transform_9_shuff, res_reference)


fig, ax = plt.subplots(1, figsize=[4,5])
Graphs.boxplot4_with_points_and_lines(res_aligned_3['res1'], res_aligned_3['res2'], res_non_aligned_3['res2'], res_aligned_shuff_3['res2'], subplot = ax, xlabel = ['Reference','Alignment-rotated', 'Random rotated', 'shuffle'], title = 'Fraction correct struggle/immobility', ylim=[-0.2,0.8])
res,res_total = st.one_way_repeated_measures_anova_general_four_groups(res_aligned_3['res1'], res_aligned_3['res2'], res_non_aligned_3['res2'], res_aligned_shuff_3['res2'])
stats_res = st.post_hoc_paired_multiple([res_aligned_3['res1'], res_aligned_3['res2'], res_aligned_3['res2']] , [res_aligned_3['res2'], res_non_aligned_3['res2'], res_aligned_shuff_3['res2']])


fig, ax = plt.subplots(1, figsize=[3,5])
Graphs.boxplot3_with_points_and_lines(res_aligned_9['res2'], res_non_aligned_9['res2'], res_aligned_shuff_9['res2'], subplot = ax, xlabel = ['Alignment-rotated', 'Random rotated', 'shuffle'], title = 'Fraction correct struggle/immobility', ylim=[-0.2,0.8])
res,res_total = st.one_way_repeated_measures_anova_general_four_groups(res_aligned_3['res1'], res_aligned_9['res2'], res_non_aligned_9['res2'], res_aligned_shuff_9['res2'])
stats_res = st.post_hoc_paired_multiple([res_aligned_3['res1'], res_aligned_9['res2'], res_aligned_9['res2']] , [res_aligned_9['res2'], res_non_aligned_9['res2'], res_aligned_shuff_9['res2']])


