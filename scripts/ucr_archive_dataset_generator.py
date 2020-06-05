#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

import numpy as np
from dovebirdia.datasets.ucr_archive_dataset import ucrArchiveDataset
from pdb import set_trace as st

dataset_params = {
    'datadir':'/home/mlweiss/Documents/wpi/research/data/ucrArchive/data/UCRArchive_2018/',
    'dataset':'Lightning7',
    'n_samples':(0,None),
    'test_size': 0.2,
    'random_state':37,
    'datasetname':'Lightning7'
}

dataset_params['dataset_name'] = '{dataset_name}_dataset'.format(dataset_name=dataset_params['datasetname'])

dataset = ucrArchiveDataset(params=dataset_params).getDataset()
