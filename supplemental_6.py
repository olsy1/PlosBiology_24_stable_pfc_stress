# -*- coding: utf-8 -*-

import numpy as np
import functions.manifold

#%% Load data
data = np.load('C:/Users/Ole/Documents/script/Analysis/Data/tst_pfc_March23.npy', allow_pickle=True).item()
res_day1, order_day1, shuff1 = manifold.manifolds(data, 'day1', 'struggle_time', 'immobility_time') 

#%% Get vector fields
dims = np.array([[0,4],[1,2],[0,1],[0,4],[0,2],[0,2]])
x_new_shift_spatial_bin, x_new_vector_spatial_bin, res_field = manifold.vector_field_spatial_bin_all(res_day1, order_day1, dims, bins = 12)

#%% Get angular trajectory
angles = manifold.angular_trajectory_all(res_day1, dims)