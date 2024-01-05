
import pingouin as pg
import pandas as pd
import numpy as N
from scipy import stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf


def mixedlm(array1, array2):
    mouse = N.hstack([N.ones(len(array1[d]))*d for d in range(len(array1))])
    values1 = N.hstack(array1)
    values2 = N.hstack(array2)

    values1 = values1-N.mean(values1)
    values1 = values1/N.std(values1)

    values2 = values2-N.mean(values2)
    values2 = values2/N.std(values2)

    df = pd.DataFrame(N.vstack((mouse, values1, values2)).T, columns=['mouse', 'values1', 'values2'])

    md = smf.mixedlm("values2 ~ values1", df, groups=df["mouse"], re_formula="~values1")
    mdf = md.fit(method=["lbfgs"])
    return mdf.summary()



def post_hoc_paired_multiple(arrays1, arrays2):
    results = {}
    for i, (array1, array2) in enumerate(zip(arrays1, arrays2)):
        _, shap1 = st.shapiro(array1)
        _, shap2 = st.shapiro(array2)

        test_func = st.ranksums if shap1 < 0.05 or shap2 < 0.05 else st.ttest_rel
        t, p = test_func(array1, array2)
        test = 'ranksum' if shap1 < 0.05 or shap2 < 0.05 else 't-test'

        p = p * len(arrays1)
        result = {'t': t, 'p': p, 'test': test}
        key = f'arrays_{i+1}'
        results[key] = result
    
    return results


# def post_hoc_paired(array1, array2, comparisons):
#     _, shap1 = st.shapiro(array1)
#     _, shap2 = st.shapiro(array2)

#     test_func = st.ranksums if shap1 < 0.05 or shap2 < 0.05 else st.ttest_rel
#     t, p = test_func(array1, array2)
#     test = 'ranksum' if shap1 < 0.05 or shap2 < 0.05 else 't-test'

#     p = p * comparisons
#     return {'t': t, 'p': p, 'test': test}



def one_way_repeated_measures_anova_general_three_groups(array1,array2,array3):
    '''
    Data arrays given as data (mice, cells) x time points
    '''
    temp=[]
    temp.extend(array1)
    temp.extend(array2)
    temp.extend(array3)
    data1=N.reshape(temp,[3,-1]).T
   
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data=data1.T
    t,p=[],[]
    sig=[]
    comp=[]
    pcrit=1-(1-0.05)**(1/len(data))
    
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1):
            local_comp=[n, i]
            comp.append(local_comp)
            tt,pp=st.ttest_rel(data[n],data[i])
            t.append(tt)
            p.append(pp)
            if pp<pcrit:
                sig.append(1)
            else:
                sig.append(0)
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'comparisons':comp}
    

    return res,res_total


def one_way_repeated_measures_anova_general_four_groups(array1,array2,array3,array4):
    '''
    Data arrays given as data (mice, cells) x time points
    '''
    temp=[]
    temp.extend(array1)
    temp.extend(array2)
    temp.extend(array3)
    temp.extend(array4)
    data1=N.reshape(temp,[4,-1]).T
   
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data=data1.T
    t,p=[],[]
    sig=[]
    comp=[]
    pcrit=1-(1-0.05)**(1/len(data))
    
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1):
            local_comp=[n, i]
            comp.append(local_comp)
            tt,pp=st.ttest_rel(data[n],data[i])
            t.append(tt)
            p.append(pp)
            if pp<pcrit:
                sig.append(1)
            else:
                sig.append(0)
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'comparisons':comp}
    

    return res,res_total


def two_way_repeated_measures_anova_general(data1,data2):
    '''
    Data arrays given as data (mice, cells) x time points
    '''
    data=data1
    data=N.append(data1,data2,axis=0)
    data=N.ravel(data)
    condition=N.zeros((len(N.ravel(data1))))      
    condition=N.append(condition,N.ones((len(N.ravel(data1)))))
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    mouse=N.append(mouse,mouse,axis=0)
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1)*2)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'condition':condition,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within=['time','condition'],subject='mouse',
                  data=df, detailed=True)
    
    
    
    return res
    

def one_way_repeated_measures_anova_general(data1):
    '''
    Data arrays given as data (mice, cells) x time points
    '''
    data=data1
    data=N.ravel(data)
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
    
    

    return res
