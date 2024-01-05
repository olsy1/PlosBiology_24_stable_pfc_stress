# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib


def plot_pie(values, subplot):
    labels = ['pl', 'cg', 'm2']
    colors = ['grey', 'white', 'yellow']
    subplot.pie(values, labels=labels, colors = colors, autopct='%1.1f%%', wedgeprops={"edgecolor":"k",'linewidth': 1,  'antialiased': True})



def boxplot1_with_points(data1,subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],showfliers=False,widths=0.5,s=30,incl_noise=True,random_range=0.1,align='center',log=False):
    
    data1=np.asarray(data1)
    data1=data1[~np.isnan(data1)]

    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    data = data1
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    subplot.set_ylim(ylim)
    x1=[]

 
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)

            
    subplot.scatter(x1,data1,c="k",s=s)

    if log==True:
        subplot.set_yscale('log')   



def boxplot2_with_points_and_lines(data1,data2,xlabel,title,  subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],showfliers=False,widths=0.5,s=30,incl_noise=True,random_range=0.1,align='center',log=False):
    
    data1=np.asarray(data1)
    data2=np.asarray(data2)
    data1=data1[~np.isnan(data1)]
    data2=data2[~np.isnan(data2)]
    
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)   
    data=[data1,data2]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    subplot.set_xticks(ticks = [1,2], labels = xlabel)
    subplot.set_title(title)
    subplot.set_ylim(ylim)

    x1=[]
    x2=[]
   
    #
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    if log==True:
        subplot.set_yscale('log')
        
def boxplot3_with_points_and_lines(data1,data2,data3,xlabel, title, subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    subplot.set_xticks(ticks = [1,2,3], labels = xlabel)
    subplot.set_title(title, fontsize = 15)
    subplot.set_ylim(ylim)
    #subplot.set_ylim(40, 200)
    #subplot.set_ylim(-0.3,1)
    #subplot.set_ylim(-0.2,0.8)
   # subplot.set_ylim(None,0.6)
    #subplot.set_ylim(0.0105, 0.016)
    #subplot.set_ylim(0.0102, 0.0145)
    #subplot.set_ylim(0.4, 1)
    #subplot.set_ylim(0.02, 0.07)
    #subplot.set_ylim(0, 1.0)
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)

    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
        _=subplot.plot([x2[n],x3[n]],[data2[n],data3[n]],"grey",alpha=0.5) 
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    subplot.scatter(x3,data3,c="k",s=s)
    

def boxplot4_with_points_and_lines(data1,data2,data3,data4, xlabel, title, subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3, data4]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    subplot.set_xticks(ticks = [1,2,3,4], labels = xlabel)
    #subplot.set_title(title, fontsize = 15)
    #subplot.set_ylim(-0.2,0.8)
    subplot.set_ylim(ylim)
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)
    for n in range(len(data4)):
        if incl_noise==True:
            x4.append(np.random.choice(np.linspace(4-random_range,4+random_range,1000)))
        else:
            x4.append(4)

    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
        _=subplot.plot([x2[n],x3[n]],[data2[n],data3[n]],"grey",alpha=0.5) 
        _=subplot.plot([x3[n],x4[n]],[data3[n],data4[n]],"grey",alpha=0.5) 
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    subplot.scatter(x3,data3,c="k",s=s)
    subplot.scatter(x4,data4,c="k",s=s)

def boxplot5_with_points_and_lines(data1,data2,data3,data4,data5, xlabel, title, subplot, figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3, data4, data5]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    subplot.set_xticks(ticks = [1,2,3,4,5], labels = xlabel)
    #subplot.set_title(title, fontsize = 15)
    subplot.set_ylim(-0.2,1)
    #subplot.set_ylim(0, 1.1)
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)
    for n in range(len(data4)):
        if incl_noise==True:
            x4.append(np.random.choice(np.linspace(4-random_range,4+random_range,1000)))
        else:
            x4.append(4)
    for n in range(len(data5)):
        if incl_noise==True:
            x5.append(np.random.choice(np.linspace(5-random_range,5+random_range,1000)))
        else:
            x5.append(5)

    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
        _=subplot.plot([x2[n],x3[n]],[data2[n],data3[n]],"grey",alpha=0.5) 
        _=subplot.plot([x3[n],x4[n]],[data3[n],data4[n]],"grey",alpha=0.5) 
        _=subplot.plot([x4[n],x5[n]],[data4[n],data5[n]],"grey",alpha=0.5) 
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    subplot.scatter(x3,data3,c="k",s=s)
    subplot.scatter(x4,data4,c="k",s=s)
    subplot.scatter(x5,data5,c="k",s=s)

def boxplot2_with_points(data1,data2,figsize=[3,5],whis=[5,95],showfliers=False,widths=0.5,s=30,incl_noise=True,random_range=0.1,align='center',log=False):
    
    data1=np.asarray(data1)
    data2=np.asarray(data2)
    data1=data1[~np.isnan(data1)]
    data2=data2[~np.isnan(data2)]
    
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=plt.subplots(figsize=figsize)   
    data=[data1,data2]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    ax.set_ylim(0, 1)
    
    x1=[]
    x2=[]
   
 
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
            
    ax.scatter(x1,data1,c="k",s=s)
    ax.scatter(x2,data2,c="k",s=s)
    if log==True:
        ax.set_yscale('log')   
    
    
def boxplot3_with_points(data1,data2,data3,figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    data1=np.asarray(data1)
    data2=np.asarray(data2)
    data3=np.asarray(data3)
    data1=data1[~np.isnan(data1)]
    data2=data2[~np.isnan(data2)]
    data3=data3[~np.isnan(data3)]
    
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]


   
 
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)


    ax.scatter(x1,data1,c="k",s=s)
    ax.scatter(x2,data2,c="k",s=s)
    ax.scatter(x3,data3,c="k",s=s)
    


def boxplot4_with_points(data1,data2,data3,data4,figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    data1=np.asarray(data1)
    data2=np.asarray(data2)
    data3=np.asarray(data3)
    data4=np.asarray(data4)
    
    data1=data1[~np.isnan(data1)]
    data2=data2[~np.isnan(data2)]
    data3=data3[~np.isnan(data3)]
    data4=data4[~np.isnan(data4)]
    
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3, data4]
    ax.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)
    for n in range(len(data4)):
        if incl_noise==True:
            x4.append(np.random.choice(np.linspace(4-random_range,4+random_range,1000)))
        else:
            x4.append(4)


    ax.scatter(x1,data1,c="k",s=s)
    ax.scatter(x2,data2,c="k",s=s)
    ax.scatter(x3,data3,c="k",s=s)
    ax.scatter(x4,data4,c="k",s=s)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/1.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def corr_matrix(data, labels):
    
    fig, ax = plt.subplots()

    im, cbar = heatmap(data, labels, labels, ax=ax,
                   cmap="hot", cbarlabel="correlations",  vmin=0, vmax=1)
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    fig.tight_layout()
    plt.show()
    
#    fig, ax = plt.subplot()
 #   ax..imshow(data,cmap='hot',interpolation='nearest')
  #  ax.set_xticks(np.arange(len(labels)), labels=labels)


def correlation_plot(data1,data2, subplot, xlim = [-1,1], ylim=[-1,1], title ='', xlabel ='',ylabel='', c='k'):
    subplot.scatter(data1, data2, s=20, facecolors = 'None', edgecolor = c)
    subplot.set_xlabel(xlabel, fontsize = 15)
    subplot.set_ylabel(ylabel, fontsize = 15)
    subplot.set_title(title, fontsize = 17)
    subplot.set_xlim(xlim)
    subplot.set_ylim(ylim) 
    #subplot.xaxis.set_tick_params(width=2, labelsize=12)
    #subplot.yaxis.set_tick_params(width=2, labelsize=12)

def sort_map(data, sort, subplot, title):
    subplot.imshow(data[sort], interpolation='nearest', origin = 'lower', extent =  (-0.5, 0.1, 0.5, -0.5), cmap='jet')
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.title.set_text(title)
    
    
# specific graphs
def predict_time(predicted, probability, bins, title, subplot):
    actual = np.arange(1,bins+1)
    predicted_prob_mean = np.mean(probability,0)
    
    predicted_mean = np.mean(predicted,0)
    error = stats.sem(predicted, nan_policy= 'omit')
    
    #fig, ax1 = plt.subplots()
    subplot.imshow(predicted_prob_mean, cmap = 'hot', origin = 'lower', extent = [1,bins, 1, bins])
    subplot.plot(actual, predicted_mean, color = 'blue')
    subplot.fill_between(np.arange(1,len(predicted_mean)+1), predicted_mean-error, predicted_mean+error,  color = 'blue', alpha=0.4)  
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.title.set_text(title)
    return 


def overlay(data1, data2, title, figsize=[30,3]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data1, color = 'k')
    ax.plot(data2, color = 'green')
    plt.title(title, size = 20)
        

def plot_trace_time(trace, time, subplot, title):
    ncell, T = trace.shape
    t = np.arange(T)/20
    for i in range(ncell):
        f = trace[i]/np.max(trace[i])
        subplot.plot(t, i+f, color='k')
    subplot.plot(t, time+i+1,color = 'green')
    #subplot.set_xticks([])
    #subplot.set_yticks([])
    subplot.title.set_text(title)
    
    
def plot_mean_manifold(res, order, subplot,dim1,dim2,dim3, s = 20):
    #fig = plt.figure(figsize=(10, 10))
    #subplot = fig.gca(projection='3d')
    #subplot.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #subplot.set_xticklabels([])
    #subplot.set_yticklabels([])
    #subplot.set_zticklabels([])

    #subplot.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #subplot.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #subplot.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #subplot.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #subplot.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #subplot.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,1),interval=100)
    #subplot.set_xlabel('Component 1', fontsize=20)
    #subplot.set_ylabel('Component 2', fontsize=20)
    #subplot.set_zlabel('Component 3', fontsize=20)
    subplot.set_axis_off()
    lc = subplot.scatter3D(res[:,dim1], res[:,dim2], res[:,dim3], c = order, cmap="RdBu", lw=2, s =s, edgecolor = 'face')
    return lc

def plot_mean_manifold2d(res, order, dim1,dim2, subplot, s = 20):
    #fig = plt.figure(figsize=(10, 10))
    #subplot = fig.gca(projection='3d')
    #subplot.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    subplot.set_xticklabels([])
    subplot.set_yticklabels([])


    #rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,1),interval=100)
    #subplot.set_xlabel('Component 1', fontsize=20)
    #subplot.set_ylabel('Component 2', fontsize=20)
    #subplot.set_zlabel('Component 3', fontsize=20)

    lc = subplot.scatter(res[:,dim1], res[:,dim2], c = order, cmap="RdBu", lw=2, s =s, edgecolor = 'face')
    subplot.set_axis_off()
    #plt.colorbar(lc)
    return lc
    


def plot_mean_manifold_single(res, order, subplot, s = 20):
    #fig = plt.figure(figsize=(10, 10))
    #subplot = fig.gca(projection='3d')
    #subplot.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #subplot.set_xticklabels([])
    #subplot.set_yticklabels([])
    #subplot.set_zticklabels([])
    subplot.set_axis_off()

    #rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,1),interval=100)
    #subplot.set_xlabel('Component 1', fontsize=20)
    #subplot.set_ylabel('Component 2', fontsize=20)
    #subplot.set_zlabel('Component 3', fontsize=20)

    lc = subplot.scatter3D(res[:,0], res[:,1], res[:,2], c = order, cmap="RdBu", lw=2, s =s, edgecolor = 'face')
    #plt.colorbar(lc)
    
    return lc


def vector_field(x_new_shift_spatial_bin,x_new_vector_spatial_bin, x_new_shift, subplot):
    
    subplot.scatter(x_new_shift[:,0], x_new_shift[:,1], c = 'grey', alpha = 0.05)
    subplot.axis('off')
    subplot.quiver(x_new_shift_spatial_bin[:,0],x_new_shift_spatial_bin[:,1],x_new_vector_spatial_bin[:,0],x_new_vector_spatial_bin[:,1], headlength = 8, headwidth = 6)
    
def visualize_single_cell(cell, behavior, title, figsize = [15,7]):
    fig, (ax1, ax2)  = plt.subplots(2,1, sharex = True, figsize=figsize)
    ax1.set_title(title)
    ax2.set_title('speed during tst')
    ax1.plot(cell, color = 'green')
    ax2.plot(behavior, color = 'k')
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    
    
def visualize_behavior(movement, struggle, immobility, figsize = [30,7]):
    struggle_movement = np.zeros(len(movement))
    immobility_movement = np.zeros(len(movement))
    for i in range(len(movement)):
        if struggle[i] ==1:
            struggle_movement[i] = movement[i]
        else:
            struggle_movement[i] = np.nan

        if immobility[i] ==1:
            immobility_movement[i] = movement[i]
        else:
            immobility_movement[i] = np.nan

    
    fig, ax1  = plt.subplots(figsize=figsize)
    ax1.plot(movement, c = 'k')
    ax1.plot(struggle_movement, c = 'blue')
    ax1.plot(immobility_movement, c='red')
    ax1.set_xticks([])
    ax1.set_yticks([])

    
def circular_hist(ax, x, bins=30, density=False, offset=0, gaps=False):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, align='edge', width=widths,
                     edgecolor='white', fill=True, linewidth=1, color='k')

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    #if density:
    ax.set_yticks([])

    return n, bins, patches


def performance_nr_neurons(res, subplot):
    x = np.arange(1, len(res[0])+1)
    
    for mouse in range(len(res)):
        subplot.plot(x, res[mouse], c = 'k', linewidth =1)
    mean = np.mean(res, axis=0)
    subplot.plot(x,mean, c = 'k', linewidth = 5)

def performance_nr_neurons_w_shuffle(res, shuffle, subplot):
    x = np.arange(1, len(res[0])+1)
    
    for mouse in range(len(res)):
        subplot.plot(x, res[mouse], c = 'k', linewidth =1)
        subplot.plot(x, shuffle[mouse], c = 'grey', linewidth = 1)
    mean = np.mean(res, axis=0)
    mean_shuffle = np.mean(shuffle, axis=0)
    subplot.plot(x,mean, c = 'k', linewidth = 5)
    subplot.plot(x,mean_shuffle, c = 'grey', linewidth = 5)
    


def dims(res, subplot):
    x = np.arange(2, len(res)+2)
    subplot.set_ylim(0.4,1)
    subplot.set_xlim(2,18)
    points = [None]*len(res[0])
    #plt.xticks(x)
    for a in range(len(res[0])):
        points[a] = np.zeros(len(res))
        for d in range(len(res)):
            points[a][d] = res[d][a]
        subplot.plot(x,points[a], c = 'k', linewidth =1)
    
    mean = np.mean(res, axis=1)
    subplot.plot(x,mean, c = 'k', linewidth = 5)
    
    return points


def dims_speed(res, subplot):
    x = np.arange(2, len(res)+2)
    subplot.set_ylim(-0.2,0.8)
    subplot.set_xlim(2,18)
    #plt.xticks(x)
    points = [None]*len(res[0]['res2'])
    for a in range(len(res[0]['res2'])):
        points[a] = np.zeros(len(res))
        for d in range(len(res)):
            points[a][d] = res[d]['res2'][a]
        subplot.plot(x,points[a], c = 'k', linewidth =1)
    
    mean = np.mean(points, axis=0)
    subplot.plot(x,mean, c = 'k', linewidth = 5)
    
    return points
        
def dims2(res1, res2, subplot):
    x = np.arange(2, len(res1)+2)
    subplot.set_ylim(0.4,1)
    subplot.set_xlim(2,18)
    plt.xticks(x)
    points1 = [None]*len(res1[0])
    points2 = [None]*len(res1[0])
    for a in range(len(res1[0])):
        points1[a] = np.zeros(len(res1))
        points2[a] = np.zeros(len(res2))
        for d in range(len(res1)):
            points1[a][d] = res1[d][a]
            points2[a][d] = res2[d][a]
        subplot.plot(x,points1[a], c = 'k', linewidth=1)
        subplot.plot(x,points2[a], c = 'grey', linewidth=1)
    
    mean1 = np.mean(res1, axis=1)
    mean2 = np.mean(res2, axis=1)
    subplot.plot(x, mean1, c = 'k', linewidth=5)
    subplot.plot(x, mean2, c = 'grey', linewidth=5)
    
    return points1, points2

def dims2_speed(res1, res2, subplot):
    x = np.arange(2, len(res1)+2)
    subplot.set_ylim(-0.2,0.8)
    subplot.set_xlim(2,18)
    plt.xticks(x)
    points1 = [None]*len(res1[0]['res'])
    points2 = [None]*len(res1[0]['res'])
    for a in range(len(res1[0]['res1'])):
        points1[a] = np.zeros(len(res1))
        points2[a] = np.zeros(len(res2))
        for d in range(len(res1)):
            points1[a][d] = res1[d]['res2'][a]
            points2[a][d] = res2[d]['res2'][a]
        subplot.plot(x,points1[a], c = 'k', linewidth=1)
        subplot.plot(x,points2[a], c = 'grey', linewidth=1)
    
    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    subplot.plot(x, mean1, c = 'k', linewidth=5)
    subplot.plot(x, mean2, c = 'grey', linewidth=5)
    
    return points1, points2
        
def dims3(res1, res2, res3, subplot):
    x = np.arange(2, len(res1)+2)
    subplot.set_ylim(0,1)
    subplot.set_xlim(2,18)
    for a in range(len(res1[0])):
        points1 = np.zeros(len(res1))
        points2 = np.zeros(len(res2))
        points3 = np.zeros(len(res3))
        for d in range(len(res1)):
            points1[d] = res1[d][a]
            points2[d] = res2[d][a]
            points3[d] = res3[d][a]
        subplot.plot(x,points1, c = 'k')
        subplot.plot(x,points2, c = 'grey')
        subplot.plot(x,points3, c = 'green')

def violin(res):
    fig=plt.figure(figsize=[3,5])
    ax=sns.violinplot(data=res,x="x",y="data", inner='quartile',cut=0, gridsize=100, scale='area', bandwidth=0.2,width=1)
    ax.set(ylim=(-1, 1))
    
    plt.show()   