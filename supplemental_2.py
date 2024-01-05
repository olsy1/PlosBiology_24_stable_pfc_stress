# -*- coding: utf-8 -*-

import numpy as np
import functions.Graphs
import matplotlib.pyplot as plt
#%% Load data
data = np.load('/datapath/dataset.npy', allow_pickle=True).item()

#%% cellreg_scores
# 294, 295, 329, 337, 339, 341
true_positive = np.array([97.57, 97.64, 92.08, 98.96, 98.59, 95.70])
true_negative = np.array([85.44, 74.13, 88.96, 93.05, 84.87, 85.76])

fig, ax = plt.subplots(figsize=(3, 6))
Graphs.boxplot1_with_points(true_negative, ax)
