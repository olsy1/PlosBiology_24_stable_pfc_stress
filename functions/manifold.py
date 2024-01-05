# -*- coding: utf-8 -*-


import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
import functions.time_functions as tf
import copy
from scipy.linalg import  svd
import functions.prediction
from scipy import stats
from typing import Optional, Tuple
import functions.rate_coding


def behavior(behavior1, behavior2): 
    behavior1 = tf.remove_edge_behavior(behavior1)
    behavior2 = tf.remove_edge_behavior(behavior2)
       
    behavior1_time = tf.make_time(behavior1, 20)
    behavior2_time = tf.make_time(behavior2, 20)
    
    order = tf.combine_time(behavior1_time, behavior2_time)
    return order

def bin_data(transients, bins=2400):
    transients_binned = stats.binned_statistic(range(len(transients[0])), transients, statistic='mean', bins=bins)[0]
    return np.transpose(transients_binned)




def embedding(transients, dim_met='Isomap'): 
    """
 	Generate the embedding of the given transients using the specified dimensionality reduction method.
 	Parameters:
 	- transients (array-like): The transients to be embedded.
 	- dim_met (str, optional): The dimensionality reduction method to be used. Defaults to 'Isomap'.
 	Returns:
 	- x_new (array-like): The embedded transients.
 	Raises:
 	- ValueError: If an invalid dimensionality reduction method is specified.
 	"""
    if dim_met == 'Isomap':
        transients_short = stats.binned_statistic(range(0,len(transients[0])), transients, statistic='mean', bins = 2400)[0]
        transients = np.transpose(transients)
        transients = tf.standardize_transients(transients, method = 'quantile') 
        transients_short = np.transpose(transients_short)
        transients_short = tf.standardize_transients(transients_short, method = 'quantile') 
        pca = Isomap(n_neighbors =500, n_components= 5) 
        pca.fit(transients_short)
        x_new = pca.transform(transients)  
    elif dim_met == 'PCA':
        transients = tf.standardize_transients(transients.T, method='quantile') 
        pca = PCA(n_components=5)
        x_new = pca.fit_transform(transients)
    elif dim_met == 'SpectralEmbedding':
        transients = tf.standardize_transients(transients.T, method ='quantile') 
        pca = SpectralEmbedding(n_neighbors =1000, n_components= 5)
        x_new = pca.fit_transform(transients)   
    return x_new

def manifolds(data, day, behav1, behav2, shuffle = True, dim_met='Isomap'): 
    """
	Generate a manifold representation of the given data.
	Parameters:
	- data: The input data to generate the manifold representation. 
	- day: The day to consider for the data. 
	- behav1: The first behavior to consider for the data. 
	- behav2: The second behavior to consider for the data. 
	- shuffle: Whether to shuffle the data or not. (type: bool, default: True)
	- dim_met: The dimensionality reduction method to use. (type: str, default: 'Isomap')
	Returns:
	- res: The manifold representation of the data. (type: list)
	- order: The ordered behaviors of the data. (type: list)
	- shuff_mani: The manifold representation of the shuffled data. (type: list)
	"""
    res = [None]*len(data)
    shuff_mani = [None]*len(data)
    order = [None]*len(data)
    a = 0
    for animal in data:
        behavior1 = copy.deepcopy(data[animal][day][behav1])
        behavior2 = copy.deepcopy(data[animal][day][behav2])
        behavior3 = copy.deepcopy(data[animal][day]['background_time'])
    
        transients = copy.deepcopy(data[animal][day]['fn_tst'])
        transients = tf.convolve(transients, 5)
        
        res[a] = embedding(transients, dim_met=dim_met)
        order[a] = behavior(behavior1, behavior2, behavior3)
        order[a] = order[a][:len(res[a])]
        
        if shuffle == True:
            shuff = prediction.neuron_shuffling_fixed_seed(transients)
            shuff_mani[a] =  embedding(shuff)
        
        a+=1
    return res, order, shuff_mani


def vector_field_spatial_bin_all(res_old,order, Dim, bins = 15): # used
    res = copy.deepcopy(res_old)
    x_new_shift_spatial_bin, x_new_vector_spatial_bin = [None] * len(res), [None] * len(res)
    for i in range(len(res)):
        dim1 = Dim[i,0]
        dim2 = Dim[i,1]
        res[i] = res[i][:,[dim1,dim2]]
        x_new_shift_spatial_bin[i], x_new_vector_spatial_bin[i] = vector_field_spatial_bin(res[i], order[i], bins = bins)
    return x_new_shift_spatial_bin, x_new_vector_spatial_bin, res
    

def vector_field_spatial_bin(x_new, order, bins = 15): 
    x_new_vector = np.zeros((len(x_new),2))
    for i in range(1,len(x_new_vector)):
     #   if order[i] !=0:
         x_new_vector[i,:] =  [x_new[i,0]-x_new[i-1,0], x_new[i,1]-x_new[i-1,1]]

    x_min = min(x_new[:,0]-0)
    x_max = max(x_new[:,0]+0)
    y_min = min(x_new[:,1]-0)
    y_max = max(x_new[:,1]+0)
    
    x_i_previous = copy.deepcopy(x_min)
    y_i_previous = copy.deepcopy(y_min)
    
    x_new_shift_spatial_bin = np.zeros((bins*bins,2))
    x_new_vector_spatial_bin = np.zeros((bins*bins,2))
    
    e = 0
    for x_i in np.arange(x_min,x_max, (x_max-x_min)/bins):
        for y_i in np.arange(y_min,y_max, (y_max-y_min)/bins):
            x_news_shift = x_new[(x_new[:,0] <= x_i) & (x_new[:,0] > x_i_previous) & (x_new[:,1] <= y_i) & (x_new[:,1] > y_i_previous)]
            x_news_vector =  x_new_vector[(x_new[:,0] <= x_i) & (x_new[:,0] > x_i_previous) & (x_new[:,1] <= y_i) & (x_new[:,1] > y_i_previous)]
            x_news_vector = x_news_vector[:,0:2]
            x_news_shift = x_news_shift[:,0:2]
            x_new_shift_spatial_bin[e] =  np.nanmean(x_news_shift,0)
            x_new_vector_spatial_bin[e] = np.nanmean(x_news_vector,0)
            y_i_previous = y_i
            e +=1
        x_i_previous = x_i 
    
    return x_new_shift_spatial_bin, x_new_vector_spatial_bin


def angular_trajectory_all(res, Dim): 
    angles = [None]*len(res)
    for i in range(len(res)):
        dim1 = Dim[i][0]
        dim2 = Dim[i][1]
        angles[i] = angular_trajectory(res[i][:,[dim1,dim2]])
    return angles

def angular_trajectory(x_new): 
    x_new = shift_2_positive(x_new)
    x_new_vector = np.zeros((len(x_new),2))
    for i in range(1,len(x_new_vector)):
        x_new_vector[i,:] =  [x_new[i,0]-x_new[i-1,0], x_new[i,1]-x_new[i-1,1]]

    x_new_shift_binned = np.transpose(x_new)
    x_new_shift_binned = stats.binned_statistic(range(0,len(x_new_shift_binned[0])), x_new_shift_binned, statistic='mean', bins = 800)[0]
    x_new_shift_binned = np.transpose(x_new_shift_binned)

 
    angle = np.empty(len(x_new_shift_binned)-2)
    for i in range(1,len(x_new_shift_binned)-1):   
        a = [x_new_shift_binned[i,0]-x_new_shift_binned[i-1,0], x_new_shift_binned[i,1]-x_new_shift_binned[i-1,1]]
        b = [x_new_shift_binned[i+1,0]-x_new_shift_binned[i,0], x_new_shift_binned[i+1,1]-x_new_shift_binned[i,1]]
        angle[i-1] = angle_between(a,b)

    return angle



def angle_between(p1,p2): 
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 -ang2) %  (2* np.pi)


def shift_2_positive(x_new):
    zero_min = np.min(x_new[:,0])
    one_min = np.min(x_new[:,1])
    x_new_shift = copy.deepcopy(x_new)
    for i in range(len(x_new)):
        if zero_min < 0:
            x_new_shift[:,0] = x_new[:,0] - zero_min
        if one_min <0:
            x_new_shift[:,1] = x_new[:,1] - one_min
    return x_new_shift
            

def manifolds_mean(data, day, behav1, behav2, manifold): 
    transients_mean = [None]*len(data)
    order = [None]*len(data)
    a = 0
    for animal in data:
        behavior1 = copy.deepcopy(data[animal][day][behav1])
        behavior2 = copy.deepcopy(data[animal][day][behav2])
        trans = np.transpose(manifold[a])
        transients_mean[a], order[a] = population_vector_mean(trans, behavior1, behavior2)
        a+=1
    return transients_mean, order

def population_vector_mean(transients, behavior1, behavior2, min_frames = 30, 
                           max_frames = 600, cut_frames = 0, bins = 20):

    transients1, time_digitized1, time_scaled1 = process(transients, behavior1, \
        min_frames = min_frames, max_frames = max_frames, \
        cut_frames = cut_frames, bins = bins)
    
    transients2, time_digitized2, time_scaled2 = process(transients, behavior2, \
        min_frames = min_frames, max_frames = max_frames, \
        cut_frames = cut_frames, bins = bins)    

    e=0
    transients_mean1 = np.zeros((len(transients1), bins))

    for i in range(1,bins+1):
        transients_mean1[:,e] = np.mean(transients1[:, time_digitized1 == i],1)
        e+=1
    
    e=0
    transients_mean2 = np.zeros((len(transients2), bins))
    for i in range(1,bins+1):
        transients_mean2[:,e] = np.mean(transients2[:, time_digitized2 == i],1)
        e+=1

    transients_mean = np.concatenate((transients_mean1, transients_mean2),1)
    order = np.concatenate((np.arange(0,bins), np.arange(0,bins)*-1))  
    
    transients_mean = np.transpose(transients_mean)
    
    return transients_mean, order

def process(transients, behavior, min_frames = 20, 
            max_frames = 1500, cut_frames = 0, bins = 60):
    
    behavior = tf.remove_edge_behavior(behavior)
    start, end = tf.start_end(behavior, min_frames, max_frames, cut_frames)
    time_scaled, transients = tf.get_relative_time(transients, start, end)
    time_digitized = np.digitize(time_scaled,np.linspace(0,1,bins))
    
    return transients, time_digitized, time_scaled

def align_manifold_days(res1, res2): 
    aligned_res = [None] *len(res1)
    b = [None] *len(res1)
    error = [None] *len(res1)
    t = [None] *len(res1)
    for i in range(len(res1)):
        aligned_res[i], b[i], error[i], t[i] = align_manifolds(res1[i],res2[i])
    return aligned_res, b, error, t


def align_manifolds(a, b): 
    a = translate_array(a)[0]
    b = translate_array(b)[0]

    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)
    a = a/a_mag
    b = b/b_mag

    if len(a) < len(b):
        a, b = pad_arrays(a, b)
    else:
        b, a = pad_arrays(b, a)

    return procrustes_os(a, b)

def pad_arrays(short_array, long_array):
    dim = len(short_array[0])
    diff = len(long_array) - len(short_array)
    short_array = np.concatenate((short_array, np.zeros((diff, dim))))
    return short_array, long_array


def procrustes_os(a,b): 
    u, _, vt = svd(np.dot(a.T, b))
    R = np.dot(u, vt)
    a_project = np.dot(a,R)
    error = 0
    return a_project, b, error, R

def scale_translate(res): 
    for i in range(len(res)):
        a = translate_array(res[i])[0]
        a_mag = np.linalg.norm(a)
        res[i] = a/a_mag
    return res



def translate_array(array_a: np.ndarray, array_b: Optional[np.ndarray] = None, weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]: 
    if weight is not None:
        if weight.ndim != 1:
            raise ValueError("The weight should be a 1d row vector.")
        if not (weight >= 0).all():
            raise ValueError("The elements of the weight should be non-negative.")

    centroid_a = np.average(array_a, axis=0, weights=weight)
    if array_b is not None:
        # translation vector to b centroid
        centroid_a -= np.average(array_b, axis=0, weights=weight)
    return array_a - centroid_a, -1 * centroid_a

def align_all_manifolds(res):
    aligned_res = [None] *len(res)
    b = [None] *len(res)
    error = [None] *len(res)
    t = [None] *len(res)
    for i in range(len(res)):
        res1 = res[i]
        res2 = res[i-1]
        aligned_res[i], b[i-1], error[i], t[i] = align_manifolds(res1,res2)
    return aligned_res, b, error,t 


def align_all_manifolds_reference(res, reference):
    b = [None] *len(res)
    error = [None] *len(res)
    t = [None] *len(res)
    for i in range(len(res)):
        res1 = res[i]
        res2 = res[reference]
        res[i], b[reference], error[i], t[i] = align_manifolds(res1,res2)
    return res, b, error,t 


def rotation(res, t, average = False, rots = 1000): 
    res_transform = [None] *len(res)
    res_non_transform = [None] *len(res)
    res_many_non_transform = [None] *rots
    for i in range(len(res)):
        translated = translate_array(res[i])[0]
        a_mag = np.linalg.norm(translated)
        a = res[i]/a_mag
        res_transform[i] = np.dot(a, t[i])
        random_ortho = stats.special_ortho_group.rvs(len(t[i]), random_state =0)
        res_non_transform[i] = np.dot(a,random_ortho)
    
    
    if average== True:
        random_states = np.arange(0,rots)
        for shuff in range(rots):
            randoms = [None]*len(res)
            shuff_nr = random_states[shuff]
            for i in range(len(res)):
                translated = translate_array(res[i])[0]#
                a_mag = np.linalg.norm(translated)
                a = res[i]/a_mag                
                random_ortho = stats.special_ortho_group.rvs(len(t[i]), random_state=shuff_nr)
                randoms[i] = np.dot(a,random_ortho)
            res_many_non_transform[shuff] = randoms
        
    return res_transform, res_non_transform, res_many_non_transform



def get_dims(data, Res, dim):
    res = [None]*len(data)
    a=0
    for animal in data:
        res[a] = Res[a][:,:dim]
        a+=1
    return res


def compare_manifold_dims(data, Res_day1, Res_day3, Res_day9, order_day1, order_day3, order_day9, Shuff1, Shuff3, Shuff9):
    dims = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20]
    
    res_manifold_aligned2_3_bin = [None]*len(dims)
    res_manifold_aligned1_3_bin = [None]*len(dims)
    res_manifold_non_aligned2_3_bin = [None]*len(dims)
    res_manifold_aligned2_9_bin = [None]*len(dims)
    res_manifold_aligned1_9_bin = [None]*len(dims)
    res_manifold_non_aligned2_9_bin = [None]*len(dims)
    res_manifold_aligned2_shuff_3_bin = [None]*len(dims)
    res_manifold_aligned2_shuff_9_bin = [None]*len(dims)
    
    res_aligned_3_speed = [None]*len(dims)
    res_non_aligned_3_speed = [None]*len(dims)
    res_aligned_shuff_3_speed = [None]*len(dims)
    res_aligned_9_speed = [None]*len(dims)
    res_non_aligned_9_speed = [None]*len(dims)
    res_aligned_shuff_9_speed = [None]*len(dims)
    res = {}
    
    d = 0
    for dim in dims:
        res_day1 = get_dims(data, Res_day1, dim)
        res_day3 = get_dims(data, Res_day3, dim)
        res_day9 = get_dims(data, Res_day9, dim)
        
        shuff3 = get_dims(data, Shuff3, dim)
        shuff9 = get_dims(data, Shuff9, dim)
        

        
        transients_mean_day1, order_mean_day1 = manifolds_mean(data,'day1', 'struggle_time', 'immobility_time', res_day1) # mean of manifolds
        transients_mean_day3, order_mean_day3 = manifolds_mean(data,'day3', 'struggle_time', 'immobility_time', res_day3)
        transients_mean_day9, order_mean_day9 = manifolds_mean(data,'day9', 'struggle_time', 'immobility_time', res_day9)

        res_align_3, b, error, t_3 = align_manifold_days(transients_mean_day3, transients_mean_day1) # align all manifold
        res_align_9, b, error, t_9 = align_manifold_days(transients_mean_day9, transients_mean_day1) # align all manifold
        res_reference = scale_translate(res_day1)# translate and scale reference matrix
        res_transform_3, res_non_transform_3, Res_many_non_transform_3 = rotation(res_day3, t_3, average = True, rots = 100) # apply rotation to full data
        res_transform_9, res_non_transform_9, Res_many_non_transform_9 = rotation(res_day9, t_9, average = True, rots = 100) 

        ################################## shuffle #################################################################
        transients_mean_day3_shuff, order_mean_day3_shuff =manifolds_mean(data,'day3', 'struggle_time', 'immobility_time', shuff3)
        transients_mean_day9_shuff, order_mean_day9_shuff = manifolds_mean(data,'day9', 'struggle_time', 'immobility_time', shuff9)
        res_align_3_shuff, b_shuff, error_shuff, t_3_shuff = align_manifold_days(transients_mean_day3_shuff, transients_mean_day1) # align all manifold
        res_align_9_shuff, b_shuff, error_shuff, t_9_shuff = align_manifold_days(transients_mean_day9_shuff, transients_mean_day1) # align all manifold
        res_transform_3_shuff, res_non_transform_3_shuff, Res_many_non_transform_3_shuff = rotation(shuff3, t_3_shuff, average = True, rots =1) # a
        res_transform_9_shuff, res_non_transform_9_shuff, Res_many_non_transform_9_shuff  = rotation(shuff9, t_9_shuff, average = True, rots=1) 

        ########################################################################################################################################
        # binary classification
        
        res_manifold_aligned2_3_bin[d], res_manifold_aligned1_3_bin[d] = rate_coding.decode_struggle_immobility_between_manifolds_days(data, 'day3', 'day1', res_transform_3, res_reference)
        res_manifold_non_aligned2_3_bin[d],_  = rate_coding.decode_struggle_immobility_between_manifolds_days(data, 'day3', 'day1', Res_many_non_transform_3,res_reference, random_rotations = True)
        res_manifold_aligned2_9_bin[d], res_manifold_aligned1_9_bin[d] = rate_coding.decode_struggle_immobility_between_manifolds_days(data, 'day9', 'day1', res_transform_9, res_reference)
        res_manifold_non_aligned2_9_bin[d],_  = rate_coding.decode_struggle_immobility_between_manifolds_days(data, 'day9', 'day1', Res_many_non_transform_9,res_reference, random_rotations = True)
        res_manifold_aligned2_shuff_3_bin[d], _ = rate_coding.decode_struggle_immobility_between_manifolds_days(data, 'day3', 'day1', res_transform_3_shuff, res_reference)
        res_manifold_aligned2_shuff_9_bin[d], _ = rate_coding.decode_struggle_immobility_between_manifolds_days(data, 'day9', 'day1', res_transform_9_shuff, res_reference)

        # speed decoding
        res_aligned_3_speed[d] = rate_coding.decode_movement_between_manifold_days(data, 'day3', 'day1', res_transform_3, res_reference)
        res_non_aligned_3_speed[d]  = rate_coding.decode_movement_between_manifold_days_rotations(data, 'day3', 'day1', Res_many_non_transform_3, res_reference)
        res_aligned_shuff_3_speed[d] = rate_coding.decode_movement_between_manifold_days(data, 'day3', 'day1', res_transform_3_shuff, res_reference)
        res_aligned_9_speed[d] = rate_coding.decode_movement_between_manifold_days(data, 'day9', 'day1', res_transform_9, res_reference)
        res_non_aligned_9_speed[d]  = rate_coding.decode_movement_between_manifold_days_rotations(data, 'day9', 'day1', Res_many_non_transform_9, res_reference)
        res_aligned_shuff_9_speed[d] = rate_coding.decode_movement_between_manifold_days(data, 'day9', 'day1', res_transform_9_shuff, res_reference)

        d+=1
        
    res['res_manifold_aligned2_3_bin'] = res_manifold_aligned2_3_bin
    res['res_manifold_aligned1_3_bin'] = res_manifold_aligned1_3_bin
    res['res_manifold_non_aligned2_3_bin'] = res_manifold_non_aligned2_3_bin
    res['res_manifold_aligned2_9_bin'] = res_manifold_aligned2_9_bin
    res['res_manifold_aligned1_9_bin'] = res_manifold_aligned1_9_bin
    res['res_manifold_non_aligned2_9_bin'] = res_manifold_non_aligned2_9_bin
    res['res_manifold_aligned2_shuff_3_bin'] = res_manifold_aligned2_shuff_3_bin
    res['res_manifold_aligned2_shuff_9_bin'] = res_manifold_aligned2_shuff_9_bin
    res['res_aligned_3_speed'] = res_aligned_3_speed
    res['res_non_aligned_3_speed'] = res_non_aligned_3_speed
    res['res_aligned_shuff_3_speed'] = res_aligned_shuff_3_speed
    res['res_aligned_9_speed'] = res_aligned_9_speed
    res['res_non_aligned_9_speed'] = res_non_aligned_9_speed
    res['res_aligned_shuff_9_speed'] = res_aligned_shuff_9_speed
        
    return res
        
        



def manifolds_unique(data, day, day_i, day_i_ref, behav1, behav2, shuffle = True):
    res = [None]*len(data)
    shuff_mani = [None]*len(data)
    order = [None]*len(data)
    nr_neurons = [None]*len(data)
    a = 0
    for animal in data:
        behavior1 = copy.deepcopy(data[animal][day][behav1])
        behavior2 = copy.deepcopy(data[animal][day][behav2])
        behavior3 = copy.deepcopy(data[animal][day]['background_time'])
        assignments = copy.deepcopy(data[animal][day]['assignments'])
        idx_filts = np.array(assignments[:,day_i] > -1) & np.array(assignments[:,day_i_ref] == -1)
        ass = assignments[idx_filts, day_i]
    
        transients = copy.deepcopy(data[animal][day]['fn_tst'])
        transients = np.vstack([data[animal][day]['fn_tst'][s] for s in ass])
        nr_neurons[a] = len(transients)
        transients = tf.convolve(transients, 5)
        
        res[a] = embedding(transients)
        order[a] = behavior(behavior1, behavior2, behavior3)
        
        if shuffle == True:
            shuff = prediction.neuron_shuffling(transients)
            shuff_mani[a] = embedding(shuff)
        else:
            shuff = 0
            shuff_mani = 0
        
        a+=1
    return res, order, shuff_mani, nr_neurons

 



