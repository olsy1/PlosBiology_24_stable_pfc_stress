#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as N
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


'''
Generalized linear model to predict calcium activity based on behavioural variables.
The data are split into training and testing datasets using n_fold folds; The Ridge model is fit to the training data and evaluated on the testing data
in each fold separately. During each iteration, the model is optimized by cross-validation on the training data over a range of alpha values defined
in alphas. The results of each fold are averaged to obtain final evaluation metrics (r2 and mean squared error).
Separate reduced models are created without individual speed or acceleration components, The ratio of the mean sqaured error
of the reduced and the full model is computed.
'''

target_dir="/data/modelling_speed_with_neuronal_signals"

### Parameters
n_splits=10 # Folds used for the cross-validation testing.
iterations=1000
random_shift_range=range(200,2400) # Random shift between 20s and 2 min
tst_threshold=0.2
baseline_threshold=-0.2

params={
        'random iterations':iterations,
        'n splits':n_splits,
        'random shift range':random_shift_range,
        'TS threshold':tst_threshold,
        'baseline threshold':baseline_threshold
        }


def scale_traces(baseline,traces):
    conc=N.append(baseline,traces,axis=1)
    scaler=MinMaxScaler()
    scaled=scaler.fit_transform(conc.T).T # Scale the conc traces
    baseline_out=scaled[:,:baseline.shape[1]]
    traces_out=scaled[:,baseline.shape[1]:]
    
    return baseline_out,traces_out


def cross_validate(X,y):
    # Define the model.
    clf=LinearRegression()
    pipe=make_pipeline(StandardScaler(), clf)
    kf=KFold(n_splits=n_splits,shuffle=True)
    
    score=[]
    y_predicted=N.zeros((len(y)))
    y_true=N.zeros((len(y)))

    for train_ind, test_ind in kf.split(X.T,y):
        X_train, X_test = X.T[train_ind], X.T[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

                
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)

        score.append(explained_variance_score(y_test,y_pred))            

        y_predicted[test_ind]=y_pred
        y_true[test_ind]=y_test
    return N.mean(score),y_true,y_predicted

def iterate_for_significance_testing(X,y,real_score):
    
   
    test_score=[]

    for i in range(iterations):
        y_new=y

        y_new=N.roll(y_new,N.random.choice(random_shift_range))
            
        temp_score,y_true,y_predicted=cross_validate(X,y_new)
                             
        test_score.append(temp_score)
    if real_score>N.percentile(test_score,95):
        sig=1
    else:
        sig=0
    #print(f'real decrease {real_decrease},test decrease {N.percentile(test_decrease,95)}')
    return sig

def fit_glm(data):
    traces=data['fn_tst'] # Get the neurons.
    baseline=data['fn_baseline']
   
    # Restrict the data to baseline or TS-active neurons, defined by a threshold.
    baseline_scaled,traces_scaled=scale_traces(baseline,traces)
    
    # Iterate over all neurons.
    score_ts,score_baseline=[],[]
    sig_ts,sig_baseline=[],[]
    y_predicted_ts_total=[]
    y_predicted_baseline_total=[]
    
    y_true_ts_total=[]
    y_true_baseline_total=[]
    for n in range(len(traces)):   # Iterate over neuron.             
        # Get the indices.
        local_ind=(N.mean(traces_scaled[n])-N.mean(baseline_scaled[n]))/(N.mean(traces_scaled[n])+N.mean(baseline_scaled[n]))
        

        # Case TS:
        if local_ind>tst_threshold:
            calcium_data=traces[n]
  
            y=calcium_data # The cell's activity to be predicted.
            X=N.empty((2,traces.shape[1]))           
            X[0]=data['tst_speed_dlc']
            X[1]=data['tst_speed_dlc']
                    
            
            # Cross-validate.
            score_local,y_true,y_predicted=cross_validate(X,y)
            score_ts.append(score_local)
            y_predicted_ts_total.append(y_predicted)
            y_true_ts_total.append(y_true)
            
            # Test significance.
            sig=iterate_for_significance_testing(X,y,score_local)
            sig_ts.append(sig)
            
        if local_ind<baseline_threshold:
            # Case baseline:
            calcium_data=baseline[n]

            
            y=calcium_data # The cell's activity to be predicted.
            X=N.empty((2,baseline.shape[1]))           
            X[0]=data['baseline_speed_dlc']
            X[1]=data['baseline_speed_dlc']   
                    
            # Cross-validate.
            score_local,y_true,y_predicted=cross_validate(X,y)
            score_baseline.append(score_local)
            y_predicted_baseline_total.append(y_predicted)
            y_true_baseline_total.append(y_true)
            
            # Test significance.
            sig=iterate_for_significance_testing(X,y,score_local)
            sig_baseline.append(sig)
            
                                                                                               
           

    res={'scores baseline':N.asarray(score_baseline),'scores ts':N.asarray(score_ts),'params':params,'sig ts':sig_ts,'sig baseline':sig_baseline,
         'y true baseline':y_true_baseline_total,'y predicted baseline':y_predicted_baseline_total,
         'y true ts':y_true_ts_total,'y predicted ts':y_predicted_ts_total}

    
    return res
    


### Executed part.


def fig2():
    animal_list=['animal294','animal295','animal329', 'animal337', 'animal339', 'animal341']
    day_list=['day1']
    #animal_list=['animal339']
    data=N.load("/mdata/tst_prefrontal_stability/data.npy",allow_pickle=True).item() # Load the entire data of all mice.
    os.chdir(target_dir)
    # Get TST data of the first possible day and fit the model.
    score_ts,score_baseline=[],[]
    score_m_ts,score_m_baseline=[],[]
    prop_ts,prop_baseline=[],[]
    y_predicted_ts_total=[]
    y_predicted_baseline_total=[]
    y_true_ts_total=[]
    y_true_baseline_total=[]
    
    res_total={}
    for day in day_list:
        score_ts,score_baseline=[],[]
        score_m_ts,score_m_baseline=[],[]
        for n in range(len(animal_list)):
            print(f'{animal_list[n]} on {day}')        
            local_data=data[animal_list[n]][day]
            res=fit_glm(local_data)
            
            score_baseline.extend(res['scores baseline'])
            score_ts.extend(res['scores ts'])
            score_m_ts.append(N.nanmean(res['scores ts']))
            score_m_baseline.append(N.nanmean(res['scores baseline']))
            prop_ts.append(N.sum(res['sig ts'])/len(res['sig ts']))
            prop_baseline.append(N.sum(res['sig baseline'])/len(res['sig baseline']))
            
            for i in range(len(res['y true ts'])):
                y_true_ts_total.append(res['y true ts'][i])
                y_predicted_ts_total.append(res['y predicted ts'][i])
            for i in range(len(res['y true baseline'])):
                y_true_baseline_total.append(res['y true baseline'][i])
                y_predicted_baseline_total.append(res['y predicted baseline'][i])
        res_day={'scores baseline':N.asarray(score_baseline),'scores ts':N.asarray(score_ts),
                 'scores baseline mouse':N.asarray(score_m_baseline),'scores ts mouse':N.asarray(score_m_ts),
                 'prop ts':N.asarray(prop_ts),'prop baseline':N.asarray(prop_baseline),
                 'y true baseline':y_true_baseline_total,'y predicted baseline':y_predicted_baseline_total,
                 'y true ts':y_true_ts_total,'y predicted ts':y_predicted_ts_total}
        res_total['%s'%day]=res_day
        
    return params
