#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

from pdb import set_trace as st

from dovebirdia.datasets.s5_dataset import s5Dataset

dataset_params = {
    'n_samples':None,
    'benchmark':'A1',
    'dataset_name':'s5_A1',
    'standardize':False,
}

dataset = s5Dataset(params=dataset_params).getDataset()
