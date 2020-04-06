#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

import numpy as np
from dovebirdia.datasets.benchmark_dataset import benchmarkDataset
import dovebirdia.stats.distributions as distributions
from pdb import set_trace as st

dataset_params = {
    'datadir':'/home/mlweiss/Documents/wpi/research/data/benchmark-4/',
    'datafile':'BearingEvents.csv',
    'datafile_true':'TruthBearingEvents.csv',
    'r':(0.0,20.0),
    'meas_dims':8,
    'datasetname':'benchmark_ALPHA',
}

dataset_params['dataset_name'] = '{dataset_name}_dataset'.format(dataset_name=dataset_params['datasetname'])

dataset = benchmarkDataset(params=dataset_params).getDataset()
