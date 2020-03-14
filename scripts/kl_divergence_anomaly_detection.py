import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from dovebirdia.utilities.base import loadDict
from dovebirdia.math.divergences import KLDivergence
from dovebirdia.math.metrics import affineInvariantDistance, logFrobeniusDistance
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pdb import set_trace as st

#########
# Dataset
#########

experiment = 'model_AEKF_train_S5_A2_OUTLIERS_false'
model = 7
results_dir = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/anomaly/{experiment}/{experiment}_model_{model}/results'.format(experiment=experiment,
                                                                                                                                                model=model)
data = loadDict(glob.glob(results_dir+'/*pkl')[0])

###########
# Variables
###########

n_trials = 1
samples = (None,None)
PLOT = False

# original measurements
phi = np.asarray([ phi for phi in data['x'] ])

# KL Divergence Variables
R = np.asarray([ kf_res['R'] for kf_res in data['kf_results'] ])

# outlier labels
labels = data['labels']

# list of confution matrices for each trial
cm_list = list()

##################
# Loop over trails
##################

for trial in np.arange(n_trials,step=1):

    # trial data
    PHI = phi[trial,samples[0]:samples[1]]
    SCALE = R[trial,samples[0]:samples[1]]
    LABELS = labels[trial,samples[0]:samples[1]]
    
    # calculate KL Divergence
    spd_dist = np.asarray([ affineInvariantDistance(SCALE[k+1],SCALE[k]) for k in np.arange(PHI.shape[0]-1) ])
    
    # scale KL Divergence
    spd_dist = MinMaxScaler().fit_transform(np.expand_dims(spd_dist,axis=-1))
    
    # hold potential outliers
    outliers = np.zeros(shape=spd_dist.shape[0]+1)
    
    # initialize outlier mode to false
    outlier_mode = False
    
    # outlier detection threshold
    threshold = 0.5
    
    # loop over kl divergence values
    for k, d in enumerate(spd_dist):
        
        # not in outlier mode
        if not outlier_mode:
            
            if d > threshold:
            
                outliers[k+1] = 1.0
                outlier_mode = True
            
        # in outlier mode
        else:
            
            if d < threshold:
                
                outliers[k+1] = 1.0
                
            else:
                
                outlier_mode = False
            
    # confusion matrix
    cm_list.append(confusion_matrix(outliers,LABELS).ravel()) 
          
    if PLOT:
        
        # figure
        plt.figure(figsize=(12,6))

        # suptitle
        plt.suptitle(trial)

        # phi
        plt.subplot(121)
        plt.plot(PHI[:],'-o',label='Phi',color='C0')
        plt.grid()
        plt.legend()

        # kl divergence
        plt.subplot(122)
        plt.plot(spd_dist,'-o',label='SPD Distance.',color='C1')
        plt.axhline(y=1,color='C2')
        plt.title('Pred: {pred}\nTrue: {true}'.format(pred=np.where(outliers==1),true=np.where(LABELS==1)))
        plt.grid()
        plt.legend()

        # show
        plt.show()
        plt.close()
    
    
############################
# Summarize Confusion Matrix
############################

cm_array = np.asarray(cm_list)
tn_total, fn_total, fp_total, tp_total = np.sum(cm_array, axis=0)

precision = tp_total / (tp_total + fp_total)
recall = tp_total / (tp_total + fn_total)
F1 = 2 * (precision*recall)/(precision+recall)

print('Experiment: {experiment}\nTP {tp_total}, FN {fn_total} FP {fp_total}\nPrecision {precision:0.2}\nRecall {recall:0.2}\nF1 {F1:0.2}\n'.format(experiment=experiment,
                                                          tp_total=tp_total,
                                                          fn_total=fn_total,
                                                          fp_total=fp_total,
                                                          precision=precision,
                                                          recall=recall,
                                                          F1=F1))


