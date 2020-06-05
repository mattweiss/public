#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

import numpy as np
from dovebirdia.datasets.drone_racing_dataset import droneRacingDataset
from pdb import set_trace as st

dataset_params = {
    'datadir':'/home/mlweiss/Documents/wpi/research/data/droneRacing/data/',
    'datafile':'outdoor_forward_5_davis_with_gt',
    'features':['tx','ty'],
    'n_samples':12000,
    'n_steps':1,
    'datasetname':'DroneALPHA'
}

dataset_params['dataset_name'] = '{dataset_name}_dataset'.format(dataset_name=dataset_params['datasetname'])

dataset = droneRacingDataset(params=dataset_params).getDataset()
