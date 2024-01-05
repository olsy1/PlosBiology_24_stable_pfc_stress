# -*- coding: utf-8 -*-

import functions.rate_coding
import numpy as np
import functions.manifold
import functions.Graphs
import matplotlib.pyplot as plt
import functions.statistics_test as st
#%% Load data
data = np.load('/datapath/dataset.npy', allow_pickle=True).item()


#%% Manifold of all mice 
res, order, shuff_mani = manifold.manifolds(data, 'day1', 'struggle_time', 'immobility_time') # embed


#%% Get mean manifolds
transients_mean, order_mean = manifold.manifolds_mean(data,'day1', 'struggle_time', 'immobility_time', res)
transients_mean_shuff, order_mean_shuff = manifold.manifolds_mean(data,'day1', 'struggle_time', 'immobility_time', shuff_mani) # shuff

#%% Allign all manifolds between mice
res_align, b, error, t = manifold.align_all_manifolds(transients_mean)
res_align_shuff, b_shuff, error_shuff, t_shuff = manifold.align_manifold_days(transients_mean_shuff, transients_mean)


#%% Apply rotation matrix to full data
res_transform, res_non_transform, Res_many_non_transform = manifold.rotation(res, t, average = True, rots = 100)
res_shuff_reference = manifold.scale_translate(res)# translate and scale reference matrix
res_transform_shuff, res_non_transform_shuff, _ = manifold.rotation(shuff_mani, t_shuff, average = True, rots =100)
res_reference = manifold.scale_translate(res)


#%% Predict struggle/immobility between mice
res_manifold_aligned2, res_manifold_aligned1 = rate_coding.decode_struggle_immobility_between_manifold2(data, 'day1', res_transform, res_reference)
res_manifold_non_aligned2  = rate_coding.decode_struggle_immobility_between_manifold2_rotations(data, 'day1', Res_many_non_transform, res_reference)

res_manifold_aligned2_shuff, _, = rate_coding.decode_struggle_immobility_between_manifold2(data, 'day1',  res_transform_shuff, res_reference)

fig, ax = plt.subplots(1, figsize=[4,5])
Graphs.boxplot4_with_points_and_lines(res_manifold_aligned1, res_manifold_aligned2, res_manifold_non_aligned2, res_manifold_aligned2_shuff, subplot = ax, xlabel = ['Reference','Alignment-rotated', 'Random rotated', 'shuffle'], title = 'Fraction correct struggle/immobility', ylim=[0.4,1])
stats_res = st.post_hoc_paired_multiple([res_manifold_aligned2, res_manifold_aligned2, res_manifold_aligned2] , [res_manifold_aligned1, res_manifold_non_aligned2, res_manifold_aligned2_shuff])

#%% Predict movement between mice
res_aligned = rate_coding.decode_movement_between_manifold(data, 'day1', res_transform, res_reference)
res_non_aligned  = rate_coding.decode_movement_between_manifold_rotations2(data, 'day1', Res_many_non_transform, res_reference)

res_aligned_shuff = rate_coding.decode_movement_between_manifold_days2(data, 'day1', 'day1', res_transform_shuff, res_shuff_reference)


fig, ax = plt.subplots(1, figsize=[4,5])
Graphs.boxplot4_with_points_and_lines(res_aligned['res1'], res_aligned['res2'], res_non_aligned['res2'],  res_aligned_shuff['res2'], subplot = ax, xlabel = ['Reference','Alignment-rotated', 'Random rotated', 'shuffle'], title = 'Fraction correct struggle/immobility', ylim=[-0.2,0.8])
stats_res = st.post_hoc_paired_multiple([res_aligned['res2'], res_aligned['res2'], res_aligned['res2']] , [res_aligned['res1'], res_non_aligned['res2'], res_aligned_shuff['res2']])
