# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import copy
import functions.prediction
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.measurements import center_of_mass
import functions.transient
import functions.time_functions as tf


def tst_v_baseline_selectivity_region(tst, baseline) :
    traces = np.concatenate((tst, baseline),1)  
    scaler = MinMaxScaler()
    traces = np.transpose(scaler.fit_transform(np.transpose(traces)))
    
    tst = traces[:,0:len(tst[0])]
    baseline = traces[:,len(tst[0]):len(baseline[0])]

    indeks = list(range(len(tst)))
    for cell in range(0,len(tst)):
        indeks[cell] = (np.mean(tst[cell])-np.mean(baseline[cell]))/(np.mean(tst[cell])+np.mean(baseline[cell]))

    return indeks


    

def neurons_per_regions(data, prl, cg, m2, day = 'day1', multisess =False):
    results = {}
    ts_selectivity_pr = [None] * len(data)
    ts_selectivity_cg = [None] * len(data)
    ts_selectivity_m2 = [None] * len(data)
    
    ts_movement_tuning_pr = [None] * len(data)
    ts_movement_tuning_cg = [None] * len(data)
    ts_movement_tuning_m2 = [None] * len(data)

    baseline_movement_tuning_pr = [None] * len(data)
    baseline_movement_tuning_cg = [None] * len(data)
    baseline_movement_tuning_m2 = [None] * len(data)
    
    pr_nr = np.zeros(len(data))
    cg_nr = np.zeros(len(data))
    m2_nr = np.zeros(len(data))
    
    a= 0
    for animal in data:
        ts_trace = data[animal][day]['fn_tst']
        baseline_trace = data[animal][day]['fn_baseline']
        ts_movement = data[animal][day]['tst_speed_dlc']
        baseline_movement = data[animal][day]['baseline_speed_dlc']
        
        
        A = data[animal][day]['A']
        dims = data[animal][day]['dims']
        ncell = data[animal][day]['ncell']
        hemisphere = data[animal][day]['hemis']
        
        cm = get_A_coordinate(A, dims, ncell)
        if multisess==True:
            ts_trace = np.vstack([ts_trace[s] for s in data[animal][day]['assignments_filts']])
            baseline_trace = np.vstack([baseline_trace[s] for s in data[animal][day]['assignments_filts']])
            
            A = data[animal]['day1']['A']
            dims = data[animal]['day1']['dims']
            ncell = data[animal]['day1']['ncell']
            cm = get_A_coordinate(A, dims, ncell)
            cm = np.vstack([cm[s] for s in data[animal]['day1']['assignments_filts']])
        
        pr_index, cg_index, m2_index = neurons_regions_index(dims, cm, prl[a], cg[a], m2[a], hemisphere)
        

        
        # if multisess ==True:
        #     assin = data[animal][day]['assignments_filts']
        #     assin_range = np.arange(0,len(ts_trace))
        #     assin_bool = [assin_range[i] in assin for i in range(len(assin_range))]
        
        #     pr_index = assin_bool and pr_index
        #     cg_index = assin_bool and cg_index
        #     m2_index = assin_bool and m2_index
        
        ts_trace_pr = ts_trace[pr_index]
        ts_trace_cg = ts_trace[cg_index]
        ts_trace_m2 = ts_trace[m2_index]
        
        baseline_trace_pr = baseline_trace[pr_index]
        baseline_trace_cg = baseline_trace[cg_index]
        baseline_trace_m2 = baseline_trace[m2_index]



        pr_nr[a] = np.sum(pr_index)
        cg_nr[a] = np.sum(cg_index)
        m2_nr[a] = np.sum(m2_index)
        
        
        if pr_nr[a] >0:
            ts_selectivity_pr[a] = tst_v_baseline_selectivity_region(ts_trace_pr, baseline_trace_pr)
            ts_movement_tuning_pr[a] = correlation(ts_trace_pr, ts_movement)
            baseline_movement_tuning_pr[a] = correlation(baseline_trace_pr, baseline_movement)
        if cg_nr[a] > 0:
            ts_selectivity_cg[a] = tst_v_baseline_selectivity_region(ts_trace_cg, baseline_trace_cg)
            ts_movement_tuning_cg[a] = correlation(ts_trace_cg, ts_movement)
            baseline_movement_tuning_cg[a] = correlation(baseline_trace_cg, baseline_movement)
        if m2_nr[a] > 0:
            ts_selectivity_m2[a] = tst_v_baseline_selectivity_region(ts_trace_m2, baseline_trace_m2)
            ts_movement_tuning_m2[a] = correlation(ts_trace_m2, ts_movement)
            baseline_movement_tuning_m2[a] = correlation(baseline_trace_m2, baseline_movement)


        
        a+=1
        
    results['ts_selectivity_pr'] = ts_selectivity_pr
    results['ts_selectivity_cg'] = ts_selectivity_cg
    results['ts_selectivity_m2'] = ts_selectivity_m2
    
    results['ts_movement_tuning_pr'] = ts_movement_tuning_pr
    results['ts_movement_tuning_cg'] = ts_movement_tuning_cg
    results['ts_movement_tuning_m2'] = ts_movement_tuning_m2

    results['baseline_movement_tuning_pr'] = baseline_movement_tuning_pr
    results['baseline_movement_tuning_cg'] = baseline_movement_tuning_cg
    results['baseline_movement_tuning_m2'] = baseline_movement_tuning_m2

    results['pr_nr'] = pr_nr
    results['cg_nr'] = cg_nr
    results['m2_nr'] = m2_nr


    return results

def neurons_regions_index(dims, cm, prl, cg, m2, hemisphere):
   prl_coor = prl * dims[1]
   cg_coor = cg * dims[1]
   m2_coor = m2 * dims[1]
   
   cmi = cm[:,1]
   
   if hemisphere == 'right':
       pr_index = cmi < prl_coor+7
       cg_index = (cmi < prl_coor+cg_coor+7) & (cmi > prl_coor-7)
       m2_index = (cmi < prl_coor+cg_coor+m2_coor+7) & (cmi > prl_coor+cg_coor-7)
   elif hemisphere == 'left':
       m2_index = cmi < m2_coor +7
       cg_index = (cmi < m2_coor+cg_coor+7) & (cmi > m2_coor-7)
       pr_index = (cmi < prl_coor+cg_coor+m2_coor+7) & (cmi > m2_coor+cg_coor-7)
    
   return pr_index, cg_index, m2_index

def get_A_coordinate(A, dims, ncell):
    cm = np.empty([ncell,2])
    for i in range(ncell):
        img = np.reshape(A[:,i].toarray(), dims, order='F')
        cm[i] = center_of_mass(img)
    return cm

def snr(data):
    d1_snr = [None] * len(data)
    d3_snr = [None] * len(data)
    d9_snr = [None] * len(data)
    
    for i, animal in enumerate(data):
        d1_snr[i] = data[animal]['day1']['snr'][data[animal]['day1']['assignments_filts']]
        d3_snr[i] = data[animal]['day3']['snr'][data[animal]['day3']['assignments_filts']]
        d9_snr[i] = data[animal]['day9']['snr'][data[animal]['day9']['assignments_filts']]
        
    return d1_snr, d3_snr, d9_snr



def activity_coping(data, day): 
    a = 0
    struggle_mean = np.empty(len(data))
    immobility_mean = np.empty(len(data))
    for animal in data:
        struggle_time = data[animal][day]['struggle_time']
        immobility_time = data[animal][day]['immobility_time']
        traces =  data[animal][day]['fn_tst']
        struggle_traces = traces[:,struggle_time==1]
        immobility_traces = traces[:,immobility_time==1]
        struggle_mean[a] = np.mean(struggle_traces)
        immobility_mean[a] = np.mean(immobility_traces)
        a+=1
    return struggle_mean, immobility_mean



# def activity_coping(data, day):
#     struggle_mean = []
#     immobility_mean = []
#     for animal_data in data.values():
#         struggle_time = animal_data[day]['struggle_time']
#         immobility_time = animal_data[day]['immobility_time']
#         traces = animal_data[day]['fn_tst']
#         struggle_traces = traces[:, np.where(struggle_time == 1)[0]]
#         immobility_traces = traces[:, np.where(immobility_time == 1)[0]]
#         struggle_mean.append(np.mean(struggle_traces))
#         immobility_mean.append(np.mean(immobility_traces))
#     return np.array(struggle_mean), np.array(immobility_mean)


def activity_during_behavior(data, day, Calcium, Behavior, transient_rate= False):
    activity = np.zeros(len(data))
    a=0
    for animal in data:
        traces = data[animal][day][Calcium]
        behavior = data[animal][day][Behavior]
        movement = data[animal][day]['tst_speed_dlc']
        
        corr = correlation(traces, movement)
        
        
        if Behavior == 'struggle_time':
            traces = traces[corr>0.2]
        elif Behavior == 'immobility_time':
            traces = traces[corr<-0.2]
        if transient_rate == True:
            traces = transient.return_transient_rate(traces, nsigma=2)
        else: 
            traces = tf.standardize_transients(traces.T, method = 'standard').T
        
        traces_events = traces[:,behavior.astype(bool)]
        activity[a] = np.mean(traces_events)
        a+=1
    return activity 
        
def behavior_performance(data):
    results = {}
    struggle_traces = [None] * len(data)
    immobility_traces = [None] * len(data)
    movement_traces = [None] * len(data)
    immobility_time = np.empty((len(data),3))
    struggle_time = np.empty((len(data),3))
    background_time = np.empty((len(data),3))
    movement_mean = np.empty((len(data),3))
    movement_auc = np.empty((len(data),3))

    for i, animal in enumerate(data):
        struggle_traces[i] = data[animal]['day1']['struggle_time']
        immobility_traces[i] = data[animal]['day1']['immobility_time']
        movement_traces[i] = data[animal]['day1']['tst_speed_dlc']
        for j, day in enumerate(data[animal]):
            immobility_time[i,j] = np.sum(data[animal][day]['immobility_time'])/20
            struggle_time[i,j] = np.sum(data[animal][day]['struggle_time'])/20
            background_time[i,j] = np.sum(data[animal][day]['background_time'])/20
            movement_mean[i,j] = np.mean(data[animal][day]['tst_speed_dlc'])
            movement_auc[i,j] = np.sum(data[animal][day]['tst_speed_dlc'])/300

    results['struggle_traces'] = struggle_traces
    results['immobility_traces'] = immobility_traces
    results['movement_traces'] = movement_traces
    results['immobility_time'] = immobility_time
    results['struggle_time'] = struggle_time
    results['background_time'] = background_time
    results['movement_mean'] = movement_mean
    results['movement_auc'] = movement_auc
    
    return results


def get_mean_speed_bodyparts(data, day, speed_trace):
    speed_bodyparts = [None] * len(data)
    a=0
    for animal in data:
        movement = data[animal][day][speed_trace]
        speed_bodyparts[a] = np.zeros(len(movement))
        b=0
        for bodypart in movement:
            speed_bodyparts[a][b] = np.mean(movement[bodypart])
            b+=1
        a+=1
    return np.array(speed_bodyparts)


def correlation_matrix_bodyparts(data, day, speed_trace):
    corr_mat = [None] * len(data)
    a=0
    for animal in data:
        movement = data[animal][day][speed_trace]
        values = [movement[each] for each in movement]
        corr_mat[a] = np.corrcoef(values)
        a+=1
    return corr_mat


def active_neurons(data):
    """
    Calculates number and fractions of neurons active/detected at all days.
    
    Parameters:
    data: data dictionary.
    
    Returns:
    dict: arrays of active neurons
    
    """
    
    nr_all_neurons = np.empty(len(data))
    d1_active = np.empty(len(data))
    d3_active = np.empty(len(data))
    d9_active = np.empty(len(data))
    always_active = np.empty(len(data))
    percentage_always_activate_d1 = np.empty(len(data))
    percentage_always_activate_d3 = np.empty(len(data))
    percentage_always_activate_d9 = np.empty(len(data))
    percentage_always_activate = np.empty(len(data))
    
    d1_only = np.empty(len(data))
    d3_only = np.empty(len(data))
    d9_only = np.empty(len(data))
    d1_and_d3_only = np.empty(len(data))
    d1_and_d9_only = np.empty(len(data))
    d3_and_d9_only = np.empty(len(data))
    d1_d3_d9_only = np.empty(len(data))
    
    for a, animal in enumerate(data):
        assign = data[animal]['day1']['assignments']
        nr_all_neurons[a] = len(assign)
        always_active[a] = len(data[animal]['day1']['assignments_filts'])
        d1_active[a] = len(data[animal]['day1']['fn_tst'])
        d3_active[a] = len(data[animal]['day3']['fn_tst'])
        d9_active[a] = len(data[animal]['day9']['fn_tst'])
        
        percentage_always_activate_d1[a] = (always_active[a]/d1_active[a]*100 if d1_active[a] != 0 else 0)
        percentage_always_activate_d3[a] = (always_active[a]/d3_active[a]*100 if d3_active[a] != 0 else 0)
        percentage_always_activate_d9[a] = (always_active[a]/d9_active[a]*100 if d9_active[a] != 0 else 0)
        percentage_always_activate[a] = (always_active[a]/nr_all_neurons[a] *100 if nr_all_neurons[a] != 0 else 0)
        
        d1_only[a] = sum(((assign[:,0] !=-1) & (assign[:,1] ==-1) & (assign[:,2] ==-1)))
        d3_only[a] = sum(((assign[:,0] ==-1) & (assign[:,1] !=-1) & (assign[:,2] ==-1)))
        d9_only[a] = sum(((assign[:,0] ==-1) & (assign[:,1] ==-1) & (assign[:,2] !=-1)))
        
        d1_and_d3_only[a] = sum(((assign[:,0] !=-1) & (assign[:,1] !=-1) & (assign[:,2] ==-1)))
        d1_and_d9_only[a] = sum(((assign[:,0] !=-1) & (assign[:,1] ==-1) & (assign[:,2] !=-1)))
        d3_and_d9_only[a] = sum(((assign[:,0] ==-1) & (assign[:,1] !=-1) & (assign[:,2] !=-1)))
        d1_d3_d9_only[a] = sum(((assign[:,0] !=-1) & (assign[:,1] !=-1) & (assign[:,2] !=-1)))
        
    results = {
        'total_neurons': nr_all_neurons,
        'd1_active': d1_active,
        'd3_active': d3_active,
        'd9_active': d9_active,
        'always_active': always_active,
        'percentage_always_activate_d1': percentage_always_activate_d1,
        'percentage_always_activate_d3': percentage_always_activate_d3,
        'percentage_always_activate_d9': percentage_always_activate_d9,
        'percentage_always_active': percentage_always_activate,
        'd1_only': d1_only,
        'd3_only': d3_only,
        'd9_only': d9_only,
        'd1_and_d3_only': d1_and_d3_only,
        'd1_and_d9_only': d1_and_d9_only,
        'd3_and_d9_only': d3_and_d9_only,
        'd1_d3_d9_only': d1_d3_d9_only
    }
    
    return results

    
def tst_v_baseline_selectivity_fractions_remaining(data):
    d1_selectivity, d3_selectivity = tst_v_baseline_selectivity_days(data, 'day1', 'day3')
    _, d9_selectivity = tst_v_baseline_selectivity_days(data, 'day1', 'day9')
    
  
    d1_sel = np.empty(len(data))
    d1_d3_sel = np.empty(len(data))
    d1_d9_sel = np.empty(len(data))
    all_sel = np.empty(len(data))
    non_sel = np.empty(len(data))
    
    for i in range(len(d1_selectivity)):
        index1 = np.absolute(d1_selectivity[i]) > 0.2
        index3 = np.absolute(d3_selectivity[i]) > 0.2
        index9 = np.absolute(d9_selectivity[i]) > 0.2
        
        index1_3 = index1 & index3 & ~index9
        index1_9 = index1 & index9 & ~index3
        index_all = index1 & index3 & index9
        index_none = index1 & ~index3 & ~index9
        
        d1_sel[i] = sum(index1)
        d1_d3_sel[i] = sum(index1_3)
        d1_d9_sel[i] = sum(index1_9)
        all_sel[i] = sum(index_all)
        non_sel[i] = sum(index_none)
    return np.vstack((d1_sel, d1_d3_sel, d1_d9_sel, all_sel, non_sel)).T
        
        
        
def selectivity_corr_selectivity_neurons(selec1, selec2):
    selec1_sel = [None] * len(selec1)
    selec2_sel = [None] * len(selec1)
    mouse_corr = [None] * len(selec1)
    for i in range(len(selec1)):
        index = np.absolute(selec1[i]) > 0.2
        selec1_sel[i] = np.array(selec1[i])[index]
        selec2_sel[i] = np.array(selec2[i])[index]
        mouse_corr[i] = stats.pearsonr(selec1_sel[i], selec2_sel[i])[0]
    return mouse_corr

def selectivity_corr(selec1, selec2):
    selec1_sel = [None] * len(selec1)
    selec2_sel = [None] * len(selec1)
    mouse_corr = [None] * len(selec1)
    for i in range(len(selec1)):
        selec1_sel[i] = np.array(selec1[i])
        selec2_sel[i] = np.array(selec2[i])
        mouse_corr[i] = stats.pearsonr(selec1_sel[i], selec2_sel[i])[0]
    return mouse_corr


def tst_v_baseline_selectivity(data, day = 'day1', transient_rate=False): 
    indeks = list(range(len(data)))
    an = 0
    for animal in data:
        baseline = data[animal][day]['fn_baseline']
        tst = data[animal][day]['fn_tst']
        
        #traces = np.concatenate((tst, baseline),1)

        if transient_rate == True:
            #traces = transient.return_transient_rate(traces, nsigma=2)
            tst = transient.return_transient_rate(tst, nsigma=3)
            baseline = transient.return_transient_rate(baseline, nsigma=3)
            
        traces = np.concatenate((tst, baseline),1)  
        scaler = MinMaxScaler()
        traces = np.transpose(scaler.fit_transform(np.transpose(traces)))
        
        tst = traces[:,0:len(tst[0])]
        baseline = traces[:,len(tst[0]):]
        
        indeks[an] = list(range(len(tst)))
        for cell in range(0,len(tst)):
            indeks[an][cell] = (np.mean(tst[cell])-np.mean(baseline[cell]))/(np.mean(tst[cell])+np.mean(baseline[cell]))
    
        an +=1
    
    return indeks


def tst_v_baseline_selectivity_days(data, day1 = 'day1', day2 = 'day3'): 
    indeks1 = []
    indeks2 = []

    for animal in data:
        baseline1 = np.vstack([data[animal][day1]['fn_baseline'][s] for s in data[animal][day1]['assignments_filts']])
        tst1 = np.vstack([data[animal][day1]['fn_tst'][s] for s in data[animal][day1]['assignments_filts']])
        
        baseline2 = np.vstack([data[animal][day2]['fn_baseline'][s] for s in data[animal][day2]['assignments_filts']])
        tst2 = np.vstack([data[animal][day2]['fn_tst'][s] for s in data[animal][day2]['assignments_filts']])
        
        traces1 = np.concatenate((tst1, baseline1), axis=1)
        scaler = MinMaxScaler()
        traces1 = scaler.fit_transform(traces1.T).T
        tst1, baseline1 = traces1[:, :len(tst1[0])], traces1[:, len(tst1[0]):]

        traces2 = np.concatenate((tst2, baseline2), axis=1)
        traces2 = scaler.fit_transform(traces2.T).T
        tst2, baseline2 = traces2[:, :len(tst2[0])], traces2[:, len(tst2[0]):]

        indeks1.append([(np.mean(t) - np.mean(b)) / (np.mean(t) + np.mean(b)) for t, b in zip(tst1, baseline1)])
        indeks2.append([(np.mean(t) - np.mean(b)) / (np.mean(t) + np.mean(b)) for t, b in zip(tst2, baseline2)])

    return indeks1, indeks2


    
    
def decode_tst_baseline_days(data, day, day_ref, model_type='LogisticRegression'): 
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    shuffl1 = np.empty((len(data)))
    shuffl2 = np.empty((len(data)))

    an = 0
    for animal in data:
        t1 = copy.deepcopy(data[animal][day]['fn_tst'])
        transients_ts = np.vstack([t1[s] for s in data[animal][day]['assignments_filts']])
        b1 = data[animal][day]['fn_baseline']
        transients_baseline = np.vstack([b1[s] for s in data[animal][day]['assignments_filts']])
        transients1 = np.hstack((transients_baseline, transients_ts))
        baseline1 = np.concatenate((np.ones(len(transients_baseline[0])), np.zeros(len(transients_ts[0]))))
        ts1 = np.concatenate((np.zeros(len(transients_baseline[0])), np.ones(len(transients_ts[0]))))
        
        t1 = copy.deepcopy(data[animal][day_ref]['fn_tst'])
        transients_ts_2 = np.vstack([t1[s] for s in data[animal][day_ref]['assignments_filts']])
        b1 = data[animal][day_ref]['fn_baseline']
        transients_baseline_2 = np.vstack([b1[s] for s in data[animal][day_ref]['assignments_filts']])
        transients2 = np.hstack((transients_baseline_2, transients_ts_2))
        baseline2 = np.concatenate((np.ones(len(transients_baseline_2[0])), np.zeros(len(transients_ts_2[0]))))
        ts2 = np.concatenate((np.zeros(len(transients_baseline_2[0])), np.ones(len(transients_ts_2[0]))))
        
        res1[an], res2[an], shuffl1[an], shuffl2[an]= prediction.decode_tst_baseline_days(transients1, transients2, ts1, ts2, baseline1, baseline2, model_type=model_type)
        an +=1
    return res1, res2, shuffl1, shuffl2

def decode_tst_baseline_days_selective_neurons(data, day, day_ref, model_type='LogisticRegression'):
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    shuffl1 = np.empty((len(data)))
    shuffl2 = np.empty((len(data)))

    selec1, _ = tst_v_baseline_selectivity_days(data, day1='day1', day2='day9')
    index = [np.absolute(s) > 0.2 for s in selec1]

    for an, animal in enumerate(data):
        t1 = copy.deepcopy(data[animal][day]['fn_tst'])
        transients_ts = np.vstack([t1[s] for s in data[animal][day]['assignments_filts']])
        b1 = data[animal][day]['fn_baseline']
        transients_baseline = np.vstack([b1[s] for s in data[animal][day]['assignments_filts']])
        transients1 = np.hstack((transients_baseline, transients_ts))[index[an]]
        baseline1 = np.concatenate((np.ones(len(transients_baseline[0])), np.zeros(len(transients_ts[0]))))
        ts1 = np.concatenate((np.zeros(len(transients_baseline[0])), np.ones(len(transients_ts[0]))))

        t1 = copy.deepcopy(data[animal][day_ref]['fn_tst'])
        transients_ts_2 = np.vstack([t1[s] for s in data[animal][day_ref]['assignments_filts']])
        b1 = data[animal][day_ref]['fn_baseline']
        transients_baseline_2 = np.vstack([b1[s] for s in data[animal][day_ref]['assignments_filts']])
        transients2 = np.hstack((transients_baseline_2, transients_ts_2))[index[an]]
        baseline2 = np.concatenate((np.ones(len(transients_baseline_2[0])), np.zeros(len(transients_ts_2[0]))))
        ts2 = np.concatenate((np.zeros(len(transients_baseline_2[0])), np.ones(len(transients_ts_2[0]))))

        res1[an], res2[an], shuffl1[an], shuffl2[an] = prediction.decode_tst_baseline_days(transients1, transients2, ts1, ts2, baseline1, baseline2, model_type=model_type)

    return res1, res2, shuffl1, shuffl2



def decode_struggle_immobility(data, mod = 'LogisticRegression'):
    res, shuff, sig = np.empty((len(data), 3)), np.empty((len(data), 3)), np.empty((len(data), 3))
    coefs = [None] * len(data)
    days = ['day1', 'day3', 'day9']

    for an, animal in enumerate(data):
        coefs[an] = [None] * 3
        for da, day in enumerate(days):
            if day in data[animal]:
                transients, immobility, movement = data[animal][day]['fn_tst'], data[animal][day]['immobility_time'], data[animal][day]['struggle_time']
                res[an, da], shuff[an, da] = prediction.decode_struggle_immobility(transients, immobility, movement, shuffle=True, mod= mod)
            else:
                res[an, da], shuff[an, da], sig[an, da] = None, None, None

    return res, shuff


def selectivity_proportions(selectivity):
    struggle_ratio = np.zeros(len(selectivity))
    immobility_ratio = np.zeros(len(selectivity))
    
    for a in range(len(selectivity)):
        struggle_ratio[a] = np.sum(selectivity[a] > 0.2)/len(selectivity[a])
        immobility_ratio[a] = np.sum(selectivity[a] < -0.2)/len(selectivity[a])
    
    return struggle_ratio, immobility_ratio
        
       

def movement_tuning(data, day, transientname, behaviorname): 
    transients, behavior, tuning = [None] * len(data), [None] * len(data), [None] * len(data)
    for a, animal in enumerate(data):
        behavior[a] = copy.deepcopy(data[animal][day][behaviorname]) 
        transients[a] = copy.deepcopy(data[animal][day][transientname])
        tuning[a] = correlation(transients[a], behavior[a])
    return tuning


def movement_tuning_first_v_second(data, day, transientname, behaviorname): 
    tuning1 = [None] * len(data)
    tuning2 = [None] * len(data)
    
    a = 0
    for animal in data:
    # process data
        behavior = copy.deepcopy(data[animal][day][behaviorname]) 
        transients = copy.deepcopy(data[animal][day][transientname])
        
        ses_len = int(len(behavior)/2)
        behavior1= behavior[:ses_len]
        behavior2= behavior[ses_len:]
        
        transients1=transients[:,:ses_len]
        transients2=transients[:,ses_len:]
        
        
        tuning1[a] = correlation(transients1, behavior1)
        tuning2[a] = correlation(transients2, behavior2)
        a+=1
    return tuning1, tuning2

def movement_tuning_across_days(data, day1, day2, transientsname, behaviorname): 
    tuning1, tuning2 = [None]*len(data), [None]*len(data)
    stability = [None]*len(data)
    
    a =0
    for animal in data:
        behavior1 = copy.deepcopy(data[animal][day1][behaviorname])
        behavior2 = copy.deepcopy(data[animal][day2][behaviorname])
        
        t1 = copy.deepcopy(data[animal][day1]['fn_tst'])
        transients1 = np.vstack([t1[s] for s in data[animal][day1]['assignments_filts']])

        t2 = copy.deepcopy(data[animal][day2]['fn_tst'])
        transients2 = np.vstack([t2[s] for s in data[animal][day2]['assignments_filts']])
        
        tuning1[a] = correlation(transients1, behavior1)
        tuning2[a] = correlation(transients2, behavior2)
        
        stability[a] = stats.pearsonr(np.nan_to_num(tuning1[a]), np.nan_to_num(tuning2[a]))[0]
        
        a+=1
    return tuning1, tuning2, stability


def get_multisess_trace(data, animal, day, cell):
    behavior = copy.deepcopy(data[animal][day]['struggle_time'])
    t1 = copy.deepcopy(data[animal][day]['fn_tst'])
    transients = np.vstack([t1[s] for s in data[animal][day]['assignments_filts']])
    cell = transients[cell]
    return cell, behavior
    

def decode_struggle_immobility_between_days(data, day, day_ref):
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    shuffl = np.empty((len(data)))

    
    an = 0
    for animal in data:
        t1 = copy.deepcopy(data[animal][day_ref]['fn_tst'])
        transients1 = np.vstack([t1[s] for s in data[animal][day_ref]['assignments_filts']])
        immobility1 = data[animal][day_ref]['immobility_time']
        struggle1 = data[animal][day_ref]['struggle_time']
        
        t2 = copy.deepcopy(data[animal][day]['fn_tst'])
        transients2 = np.vstack([t2[s] for s in data[animal][day]['assignments_filts']])
        immobility2 = data[animal][day]['immobility_time']
        struggle2 = data[animal][day]['struggle_time']
        
        shuff_data = prediction.neuron_shuffling_fixed_seed(transients2)
        
        res1[an], res2[an] = prediction.between_binary(np.transpose(transients1), np.transpose(transients2), struggle1, struggle2, immobility1, immobility2, stand_method='standard', C=0.0002)
        _, shuffl[an] = prediction.between_binary(np.transpose(transients1), np.transpose(shuff_data), struggle1, struggle2, immobility1, immobility2, stand_method='standard', C=0.0002)
        an +=1
    return res2, res1, shuffl



def decode_speed_between_days(data, day, day_ref):
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    shuffl = np.empty((len(data)))

    
    an = 0
    for animal in data:
        t1 = copy.deepcopy(data[animal][day_ref]['fn_tst'])
        transients1 = np.vstack([t1[s] for s in data[animal][day_ref]['assignments_filts']])
        movement1 = data[animal][day_ref]['tst_speed_dlc']
          
        t2 = copy.deepcopy(data[animal][day]['fn_tst'])
        transients2 = np.vstack([t2[s] for s in data[animal][day]['assignments_filts']])
        movement2 = data[animal][day]['tst_speed_dlc']
        
        shuff_data = prediction.neuron_shuffling_fixed_seed(transients2)
        
        res1[an], res2[an], _, _, _, _ = prediction.between_continous(np.transpose(transients1),  movement1, np.transpose(transients2), movement2, stand_method='standard')
        _, shuffl[an], _, _, _ , _ = prediction.between_continous(np.transpose(transients1), movement1, np.transpose(shuff_data), movement2, stand_method='standard')
        

        an +=1
    return res2, res1, shuffl


def decode_struggle_immobility_between_manifolds_days2(data, day, day_ref, manifold, manifold_ref): 
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    
    an = 0
    for animal in data:
        transients1 = manifold_ref[an]
        immobility1 = data[animal][day_ref]['immobility_time']
        struggle1 = data[animal][day_ref]['struggle_time']

        transients2 = manifold[an]
        immobility2 = data[animal][day]['immobility_time']
        struggle2 = data[animal][day]['struggle_time']
        

        res1[an], res2[an] = prediction.between_binary(transients1, transients2, struggle1, struggle2, immobility1, immobility2)
        an+=1    
        
    return res2, res1


def decode_movement_between_manifold_rotations2(data, day, manifold, manifold2):
    results = {}
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    predict2 = [None] * len(data)
    outcome2 = [None] * len(data)
    
    res1_all, res2_all = [None]*len(manifold), [None]*len(manifold)
    for shuffl in range(len(manifold)):
        res1_all[shuffl], res2_all[shuffl] = [None]*len(data), [None]*len(data)
        an = 0
        animal_name = list(data.keys())
        for animal in data:
            transients1 = manifold2[an-1]        
            mov1 = data[animal_name[an-1]][day]['tst_speed_dlc']
                    
            transients2 = manifold[shuffl][an]
            mov2 = data[animal_name[an]][day]['tst_speed_dlc']
            
            res1_all[shuffl][an], res2_all[shuffl][an], predict2[an], outcome2[an],_,_ = prediction.between_continous(transients1, mov1, transients2, mov2)
            an +=1
    res1 = np.mean(res1_all, 0)
    res2 = np.mean(res2_all, 0)
    res1_absolute = np.max(np.absolute(res1_all), 0)
    res2_absolute = np.max(np.absolute(res2_all), 0)
    
    results['res1'] = res1
    results['res2'] = res2
    results['predict2'] = predict2
    results['outcome2'] = outcome2
    results['res1_all'] = res1_all
    results['res2_all'] = res2_all
    results['res1_absolute'] = res1_absolute
    results['res2_absolute'] = res2_absolute
    return results


def decode_movement_between_manifold_days2(data, day, day_ref, manifold, manifold_ref): 
    results = {}
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    predict2 = [None] * len(data)
    outcome2 = [None] * len(data)
    
    an = 0
    for animal in data:
        transients1 = manifold_ref[an]        
        mov1 = data[animal][day_ref]['tst_speed_dlc']     
        transients2 = manifold[an]
        mov2 = data[animal][day]['tst_speed_dlc']
        res1[an], res2[an], predict2[an], outcome2[an],_,_ = prediction.between_continous(transients1, mov1, transients2, mov2)
        an +=1
    
    results['res1'] = res1
    results['res2'] = res2
    results['predict2'] = predict2
    results['outcome2'] = outcome2
    return results


def decode_movement_between_manifold_days_rotations2(data, day, day_ref, manifold, manifold_ref): 
    results = {}
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    predict2 = [None] * len(data)
    outcome2 = [None] * len(data)

    res1_all, res2_all = [None]*len(manifold), [None]*len(manifold)
    for shuffl in range(len(manifold)):
        res1_all[shuffl], res2_all[shuffl] = [None]*len(data), [None]*len(data)
        an = 0
        for animal in data:
            transients1 = manifold_ref[an]        
            mov1 = data[animal][day_ref]['tst_speed_dlc']
            
            imm1 = data[animal][day_ref]['struggle_time']        
            transients1 = transients1[imm1==1]
            mov1 = mov1[imm1==1]
             
            transients2 = manifold[shuffl][an]
            mov2 = data[animal][day]['tst_speed_dlc']
            
            imm2 = data[animal][day]['struggle_time']
            transients2 = transients2[imm2==1]
            mov2 = mov2[imm2==1]
        
            res1_all[shuffl][an], res2_all[shuffl][an], predict2[an], outcome2[an],_,_ = prediction.between_continous(transients1, mov1, transients2, mov2)
            an +=1
    res1 = np.mean(res1_all, 0)
    res2 = np.mean(res2_all, 0)
    
    results['res1'] = res1
    results['res2'] = res2
    results['predict2'] = predict2
    results['outcome2'] = outcome2
    return results


# def decode_struggle_immobility_between_manifold2(data, day, manifold, manifold2): 
#     res1 = np.empty((len(data)))
#     res2 = np.empty((len(data)))
    
#     an = 0
#     animal_name = list(data.keys())
#     for animal in data:
#         transients1 = manifold2[an-1]
#         immobility1 = data[animal_name[an-1]][day]['immobility_time']
#         struggle1 = data[animal_name[an-1]][day]['struggle_time']
    
#         transients2 = manifold[an]
#         immobility2 = data[animal][day]['immobility_time']
#         struggle2 = data[animal][day]['struggle_time']
#         res1[an], res2[an]= prediction.between_binary(transients1, transients2, struggle1, struggle2, immobility1, immobility2)
#         an +=1
    
#     return res2, res1

def decode_struggle_immobility_between_manifold2(data, day, manifold, manifold2): 
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    
    animal_name = list(data.keys())
    for an, animal in enumerate(data):
        transients1 = manifold2[an-1]
        immobility1 = data[animal_name[an-1]][day]['immobility_time']
        struggle1 = data[animal_name[an-1]][day]['struggle_time']
    
        transients2 = manifold[an]
        immobility2 = data[animal][day]['immobility_time']
        struggle2 = data[animal][day]['struggle_time']
        res1[an], res2[an]= prediction.between_binary(transients1, transients2, struggle1, struggle2, immobility1, immobility2)
    
    return res2, res1


def decode_struggle_immobility_between_manifolds_days2_rotations(data, day, day_ref, manifold, manifold_ref): 
    res2 = np.empty((len(data)))
        
    many_res = [None] *len(manifold)
    for shuff in range(len(manifold)):
        many_res[shuff] = np.empty((len(data)))
        an = 0
        for animal in data:
            transients1 = manifold_ref[an]
            immobility1 = data[animal][day_ref]['immobility_time']
            struggle1 = data[animal][day_ref]['struggle_time']
            
            immobility2 = data[animal][day]['immobility_time']
            struggle2 = data[animal][day]['struggle_time']
            transients2 = manifold[shuff][an]
            _,many_res[shuff][an] = prediction.between_binary(transients1, transients2, struggle1, struggle2, immobility1, immobility2)
            an+=1
    res2 = np.mean(many_res, 0)
   
    return res2


def decode_struggle_immobility_between_manifold2_rotations(data, day, manifold, manifold2): 
    res2 = np.empty((len(data)))

    animal_name = list(data.keys())
    many_res = [None]*len(manifold)
    for shuff in range(len(manifold)):
        many_res[shuff] = np.empty((len(data)))

        an = 0
        for animal in data:
            transients1 = manifold2[an-1]
            struggle1 = data[animal_name[an-1]][day]['struggle_time']
            immobility1 = data[animal_name[an-1]][day]['immobility_time']
            
            immobility2 = data[animal][day]['immobility_time']
            struggle2 = data[animal][day]['struggle_time']
            transients2 = manifold[shuff][an]
            _,many_res[shuff][an]= prediction.between_binary(transients1, transients2, struggle1, struggle2, immobility1, immobility2)
            an+=1
    res2 = np.mean(many_res,0)

    return res2


def decode_movement_between_manifold(data, day, manifold, manifold2): 
    results = {}
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    predict2 = [None] * len(data)
    outcome2 = [None] * len(data)
    
    an = 0
    animal_name = list(data.keys())
    for animal in data:
        transients1 = manifold2[an-1]        
        mov1 = data[animal_name[an-1]][day]['tst_speed_dlc']
                
        transients2 = manifold[an]
        mov2 = data[animal_name[an]][day]['tst_speed_dlc']
        
        res1[an], res2[an], predict2[an], outcome2[an],_,_ = prediction.between_continous(transients1, mov1, transients2, mov2)
        an +=1
    
    results['res1'] = res1
    results['res2'] = res2
    results['predict2'] = predict2
    results['outcome2'] = outcome2
    return results

        
def correlation(trace1, trace2): 
    """
    Calculate the correlation between two sets of traces.
    Parameters:
    - trace1 (numpy.ndarray): The first set of traces.
    - trace2 (numpy.ndarray): The second set of traces.
    Returns:
    - cor (numpy.ndarray): An array containing the correlation coefficients between the corresponding traces in trace1 and trace2.
    """
    cor = np.empty(len(trace1))
    for i in range(len(trace1)):
        cor[i] = stats.pearsonr(np.nan_to_num(trace1[i]), np.nan_to_num(trace2))[0]
    return cor


def decode_movement_between_tst_baseline(data, day, pred_from = 'baseline'): 
    results = {}
    res1 = np.empty((len(data)))
    res2 = np.empty((len(data)))
    predict2 = [None] * len(data)
    outcome2 = [None] * len(data)
    predict_within = [None] * len(data)
    outcome_within = [None] * len(data)
    
    within_shuff = np.empty((len(data)))
    between_shuff = np.empty((len(data)))
    
    animal_name = list(data.keys())
    for an, animal in enumerate(data):
        #'tst_speed_bodyparts'
        # ts label = ['camera', 'neck', 'leftfoot', 'rightfoot']
        
        #'baseline_speed_bodyparts'
        # baseline label =  ['nose', 'neck', 'leftbody', 'rightbody', 'basetail']
        
        transients1 = data[animal][day]['fn_tst']  
        mov1 = data[animal][day]['tst_speed_dlc']
        #mov1 = data[animal][day]['tst_speed_bodyparts']['rightfoot']
        
    
        transients2 = data[animal][day]['fn_baseline']
        mov2 = data[animal_name[an]][day]['baseline_speed_dlc']
        #mov2 = data[animal_name[an]][day]['baseline_speed_bodyparts']['basetail']
        
        transients1_shuff = prediction.neuron_shuffling_fixed_seed(transients1)
        transients2_shuff = prediction.neuron_shuffling_fixed_seed(transients2)
        
        transients1 = np.transpose(transients1)
        transients2 = np.transpose(transients2)
        transients1_shuff = np.transpose(transients1_shuff)
        transients2_shuff = np.transpose(transients2_shuff)
        
        if pred_from == 'baseline':
            res1[an], res2[an], predict2[an], outcome2[an], predict_within[an], outcome_within[an] = prediction.between_continous(transients2, mov2, transients1, mov1)
            within_shuff[an], between_shuff[an], _, _,_,_ = prediction.between_continous(transients2_shuff, mov2, transients1_shuff, mov1)
            
        elif pred_from == 'tst':
            res1[an], res2[an], predict2[an], outcome2[an], predict_within[an], outcome_within[an] = prediction.between_continous(transients1, mov1, transients2, mov2)
            within_shuff[an], between_shuff[an], _, _, _, _ = prediction.between_continous(transients1_shuff, mov1, transients2_shuff, mov2)
    
    shuff = np.mean((within_shuff, between_shuff),0)
    
    results['within_res'] = res1
    results['between_res'] = res2
    results['predict2'] = predict2
    results['outcome2'] = outcome2
    results['shuff'] = shuff
    results['within_shuff'] = within_shuff
    results['between_shuff'] = between_shuff
    results['predict_within'] = predict_within
    results['outcome_within'] = outcome_within
    
    return results#res1, within_shuff#results


# def standardize_transients(transients):
#     """
#     Standardizes the given transients using the StandardScaler.
#     Parameters:
#         transients (array-like): The transients to be standardized.
#     Returns:
#         array-like: The standardized transients.
#     """
#     scaler = preprocessing.StandardScaler().fit(transients)
#     transient_scaled = scaler.transform(transients)
#     return transient_scaled




    
