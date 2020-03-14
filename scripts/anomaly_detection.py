#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:0

import os
import socket
import numpy as np
import glob
import argparse
import dill
from datetime import datetime
from time import time
import csv

from pdb import set_trace as st

if socket.gethostname() != 'pengy':
    
    import matplotlib
    matplotlib.use('Agg') 
    
import matplotlib.pyplot as plt

from collections import OrderedDict

from dovebirdia.utilities.base import loadDict
from dovebirdia.math.divergences import KLDivergence, vonNeumannEntropyDivergence, logDetDivergence
from dovebirdia.math.metrics import affineInvariantDistance, logFrobeniusDistance

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pdb import set_trace as st

################################################################################
# PROCESS COMMAND LINE FLAGS
################################################################################

# define command line parser
parser = argparse.ArgumentParser()

# get train/test flag
parser.add_argument("-c", "--config", dest = "config", help="Configuration File")
parser.add_argument("-r", "--results", dest = "results", help="Results q File Description")

# parse options from commandline and copy to dictionary
flag_dict = parser.parse_args().__dict__

# read flag_dict values
config_dir = flag_dict['config']

# read all config files
config_files = os.listdir(config_dir)

config_dicts = dict()

for config_file in config_files:

    config_name = os.path.splitext(config_file)[0].split('_')[-1]

    with open(config_dir + config_file, 'rb') as handle:

        config_dicts[config_name] = dill.load(handle)

# create results directory
current_time = datetime.now().strftime("%m%d-%H-%M-%S")
current_day = datetime.now().strftime("%m%d") + '/'
res_dir = flag_dict['results']

res_file = res_dir.split('/')[-2]
        
#########
# Dataset
#########

experiment = config_dicts['model']['experiment']
model = config_dicts['model']['model']
data_dir = config_dicts['model']['data_dir'] + '{experiment}/{experiment}_model_{model}/results'.format(experiment=experiment, model=model)
data = loadDict(glob.glob(data_dir+'/*pkl')[0])

###########
# Variables
###########

n_trials = config_dicts['model']['n_trials']
samples = config_dicts['model']['samples']
PLOT = config_dicts['model']['plot']
step = config_dicts['model']['step']

# original measurements
phi = np.asarray([ phi for phi in data['x'] ])

# Covariance Matrices
R = np.asarray([ kf_res['R'] for kf_res in data['kf_results'] ])

for trial in np.arange(R.shape[0]):

    for sample in np.arange(R.shape[1]):

        #R[trial,sample] = StandardScaler().fit_transform(R[trial,sample])
        R[trial,sample] /= R[trial,sample].max()
        
# outlier labels
labels = data['labels']

# list of confution matrices for each trial
cm_list = list()

##################
# Loop over trails
##################

start_time = time()

# set trials if None
n_trials = phi.shape[0] if n_trials is None else n_trials

for trial in np.arange(n_trials,step=step):

    print('Computing trial {trial} results'.format(trial=trial+1))
    
    # list to hold SPD matrix distances and their sequential rations, will be cast to numpy arrays
    spd_dist = list()
    
    # trial data
    PHI = phi[trial,samples[0]:samples[1]]
    SCALE = R[trial,samples[0]:samples[1]]
    LABELS = labels[trial,samples[0]:samples[1]]

    # basis for comparison, initialize to first value in spd_dist
    SCALE_OLD_INDEX = 0
    SCALE_OLD = SCALE[SCALE_OLD_INDEX]

    # hold potential outliers
    outliers = np.zeros(shape=PHI.shape[0],dtype=np.int32)

    # outlier detection threshold
    threshold = config_dicts['model']['threshold']
    
    # loop over R-values
    for k in np.arange(PHI.shape[0]):

        # set SCALE_NEW
        if config_dicts['model']['metric'].__name__ == 'KLDivergence':

            SCALE_NEW = (np.zeros(SCALE[k].shape[0]),SCALE[k])
            SCALE_OLD = (np.zeros(SCALE_OLD.shape[0]),SCALE_OLD)
            
        else:

            SCALE_NEW = SCALE[k]
            
        # append SPD distance calculation
        spd_dist_k = config_dicts['model']['metric'](SCALE_NEW,SCALE_OLD)
        
        # append to SPD distance list
        spd_dist.append(spd_dist_k)


        # Test for outlier
        if spd_dist[-1] > config_dicts['model']['threshold']:

            outliers[k] = 1
            
        else:

            SCALE_OLD = SCALE_NEW
            SCALE_OLD_INDEX = k
        
    # cast to numpy array
    spd_dist = np.asarray(spd_dist)

    # confusion matrix
    cm_list.append(confusion_matrix(outliers,LABELS).ravel()) 

    if PLOT:
        
        # figure
        plt.figure(figsize=(12,6))
        
        # suptitle
        #plt.suptitle(trial)

        # phi
        plt.subplot(121)
        plt.plot(PHI[:],'-o',label='Phi',color='C0')
        plt.grid()
        #plt.legend()
        plt.xlabel('Sample Index')
        plt.ylabel('Synthetic Response')
        
        #max_scale_eigval = np.asarray([ np.linalg.eigvalsh(r) for r in SCALE ]).max(axis=1)
        max_fro_norm = np.asarray([ np.linalg.norm(r,ord='fro') for r in SCALE ])
        matrix_metric = max_fro_norm
        
        # SPD Distancce
        plt.subplot(122)
        plt.plot(spd_dist,'-o',label='SPD Dist.',color='C1')
        #plt.axhline(y=threshold,color='C2')
        #plt.title('Pred: {pred}\nTrue: {true}'.format(pred=np.where(outliers==1),true=np.where(LABELS==1)))
        plt.xlabel('Sample Index')
        plt.ylabel('Matrix Distance')
        plt.grid()
        #plt.legend()

        # show
        plt.savefig(os.getcwd() + config_dicts['model']['results_dir'] + 'figures/phi_spd_dist_plot_{trial}'.format(trial=trial+1), dpi=150)
        #plt.show()
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

metric_results_dict = OrderedDict()
metric_results_dict['tn'] = tn_total
metric_results_dict['fn'] = fn_total
metric_results_dict['fp'] = fp_total
metric_results_dict['tp'] = tp_total
metric_results_dict['precision'] = precision
metric_results_dict['recall'] = recall
metric_results_dict['F1'] = F1
metric_results_dict['runtime'] = (time() - start_time) / 60.0

################################################################################
# CSV
################################################################################

# merge dictionaries in config_dicts and training_results_dict
merged_config_dicts = OrderedDict()

# metric results
merged_config_dicts.update(metric_results_dict)

# config dictionaries
for config_dict in config_dicts.values():

    merged_config_dicts.update(config_dict)

# model id
merged_config_dicts.update({'model_id':os.getcwd().split('/')[-1].split('_')[-1]})

# change dictionary value to name if exists
for k,v in merged_config_dicts.items():

    try:

        merged_config_dicts[k] = v.__name__

    except:

        pass

results_file = os.getcwd().split('/')[-1] + '_testing_results.csv'
results_file_path = os.getcwd() + config_dicts['model']['results_dir'] + results_file

try:

    with open(results_file_path, 'a') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=merged_config_dicts.keys())

        if os.stat(results_file_path).st_size == 0:

            writer.writeheader()

        writer.writerow(merged_config_dicts)

except IOError:

    print("I/O error")
