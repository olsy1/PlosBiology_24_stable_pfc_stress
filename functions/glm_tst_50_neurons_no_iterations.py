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

target_dir="/data"

### Parameters
n_splits=10 # Folds used for the cross-validation testing.
tst_threshold=0.2
baseline_threshold=-0.2
neuron_number=50
iterations=100
convolve_sd=500
random_shift_range=range(200,2400) # Random shift between 20s and 2 min

params={'tst threshold':tst_threshold,
        'random iterations':iterations,
        'n splits':n_splits,
        'convolve sd':convolve_sd,
        'random shift range':random_shift_range
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


def fit_glm(data):
    traces=data['fn_tst'] # Get the neurons.
    baseline=data['fn_baseline']
    
    # Iterate over all neurons.
    print(len(traces),'neurons found..')
    print("Iterating over neurons..")
    score_ts,score_ts_rand=[],[]
    score_baseline,score_baseline_rand=[],[]


    y_predicted_total_ts,y_true_total_ts=[],[]
    y_predicted_total_baseline,y_true_total_baseline=[],[]
    
    
    baseline_scaled,traces_scaled=scale_traces(baseline,traces)
      
    for n in range(len(traces)):   # Iterate over neuron.             
        if n==10 or n==50 or n==100 or n==150:
            print(n)
        
        
        # Get the indices.
        local_ind=(N.mean(traces_scaled[n])-N.mean(baseline_scaled[n]))/(N.mean(traces_scaled[n])+N.mean(baseline_scaled[n]))
        
        if local_ind>tst_threshold:
            y=traces[n] # The cell's activity to be predicted.
            # Construct X from the activity of other neurons.
            other_neurons=N.delete(traces,n,axis=0)
            rand_ind=N.random.choice(len(other_neurons),neuron_number,replace=False)
            X=other_neurons[rand_ind,:]
        
            # Cross-validate with the full model.
            score_local,y_true,y_predicted=cross_validate(X,y)
            score_ts.append(score_local)
            y_predicted_total_ts.extend(y_predicted)
            y_true_total_ts.extend(y_true)
            
            
            y_shifted=N.roll(y,N.random.choice(random_shift_range))
            score_local,y_true,y_predicted=cross_validate(X,y_shifted)
            score_ts_rand.append(score_local)

        
        elif local_ind<baseline_threshold:
            y=baseline[n]
            # Construct X from the activity of other neurons.
            other_neurons=N.delete(baseline,n,axis=0)
            rand_ind=N.random.choice(len(other_neurons),neuron_number,replace=False)
            X=other_neurons[rand_ind,:]
        
            # Cross-validate with the full model.
            score_local,y_true,y_predicted=cross_validate(X,y)
            score_baseline.append(score_local)
            y_predicted_total_baseline.extend(y_predicted)
            y_true_total_baseline.extend(y_true)
            
            
            y_shifted=N.roll(y,N.random.choice(random_shift_range))
            score_local,y_true,y_predicted=cross_validate(X,y_shifted)
            score_baseline_rand.append(score_local)
            
            
        

    res={'scores ts':N.asarray(score_ts),'scores ts random':N.asarray(score_ts_rand),'params':params,         
         'y pred ts':N.reshape(y_predicted_total_ts,[len(score_ts),-1]),'y true ts':N.reshape(y_true_total_ts,[len(score_ts),-1]),
         'scores baseline':N.asarray(score_baseline),'scores baseline random':N.asarray(score_baseline_rand),
         'y pred baseline':N.reshape(y_predicted_total_baseline,[len(score_baseline),-1]),'y true baseline':N.reshape(y_true_total_baseline,[len(score_baseline),-1]),
         }

    
    return res
    
### Executed part.

def supplemental4():
    animal_list=['animal294','animal295','animal329', 'animal337', 'animal339', 'animal341']
    data=N.load("data/data.npy",allow_pickle=True).item() # Load the entire data of all mice.
    os.chdir(target_dir)
    # Get TST data of the first possible day and fit the model.    
    y_predicted_total_ts,y_true_total_ts=[],[]
    y_predicted_total_baseline,y_true_total_baseline=[],[]
    
    score_ts,score_ts_rand=[],[]
    score_baseline,score_baseline_rand=[],[]
    
    m_scores_ts,m_scores_ts_random=[],[]
    m_scores_baseline,m_scores_baseline_random=[],[]
    for n in range(len(animal_list)):
    
        local_data=data[animal_list[n]]['day1']
        
        print(animal_list[n])
    
        res=fit_glm(local_data)
        
        score_ts.extend(res['scores ts'])
        score_ts_rand.extend(res['scores ts random'])
        score_baseline.extend(res['scores baseline'])
        score_baseline_rand.extend(res['scores baseline random'])
        for i in range(len(res['y pred ts'])):
            y_predicted_total_ts.append(res['y pred ts'][i])
            y_true_total_ts.append(res['y true ts'][i])
        for i in range(len(res['y pred baseline'])):
            y_predicted_total_baseline.append(res['y pred baseline'][i])
            y_true_total_baseline.append(res['y true baseline'][i])
        
        m_scores_ts.append(N.nanmean(res['scores ts']))
        m_scores_ts_random.append(N.nanmean(res['scores ts random']))
        m_scores_baseline.append(N.nanmean(res['scores baseline']))
        m_scores_baseline_random.append(N.nanmean(res['scores baseline random']))
    
    m_scores_ts=N.asarray(m_scores_ts)
    m_scores_ts_random=N.asarray(m_scores_ts_random)
    m_scores_baseline=N.asarray(m_scores_baseline)
    m_scores_baseline_random=N.asarray(m_scores_baseline_random)
    res_total={'scores ts':N.asarray(score_ts),'scores ts random':N.asarray(score_ts_rand),'params':params,         
         'y pred ts':y_predicted_total_ts,'y true ts':y_true_total_ts,
         'scores baseline':N.asarray(score_baseline),'scores baseline random':N.asarray(score_baseline_rand),
         'y pred baseline':y_predicted_total_baseline,'y true baseline':y_true_total_baseline,
         'score ts mouse':m_scores_ts,'scores ts mouse random':m_scores_ts_random,
         'score baseline mouse':m_scores_baseline,'scores baseline mouse random':m_scores_baseline_random,
         }
    
    # Calculate stats
    return res_total

