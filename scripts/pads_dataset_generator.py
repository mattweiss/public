#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

import numpy as np
from dovebirdia.datasets.pads_dataset import padsDataset
import dovebirdia.stats.distributions as distributions
from pdb import set_trace as st

dataset_params = {
    'datadir':'/home/mlweiss/Documents/wpi/research/data/pads/',
    'datafile':'ConfLongDemo_JSI.txt',
    'datasetname':'pads_walking_BETA',
    'activity':'walking',
    'min_len':50, # ignore datasets less than this
    'max_len':100, # break datasets into subsets of this length
    'standardize':True,
    'begin_pad':10, # number of leading terms to exclude from possible missing data
    'noise':['none'],
    #'noise':['gaussian', np.random.normal, {'loc':0.0, 'scale':0.05}],
    #'noise':['bimodal', distributions.bimodal, {'loc1':0.05, 'scale1':0.03, 'loc2':-0.05, 'scale2':0.03}],
    #'noise':['cauchy', distributions.stable, {'alpha':(1.0), 'scale':0.002}],
    'mask_percent':5,
    'mask_value':1000.0,
}

dataset_params['dataset_name'] = '{dataset_name}_dataset_NOISE_{noise}_MASK_percent_{mask_percent}_value_{mask_value}'.format(dataset_name=dataset_params['datasetname'],
                                                                                                                                           noise=dataset_params['noise'][0],
                                                                                                                                           mask_percent=str(dataset_params['mask_percent']).replace('.','-'),
                                                                                                                                           mask_value=str(int(dataset_params['mask_value'])))

dataset = padsDataset(params=dataset_params).getDataset()
