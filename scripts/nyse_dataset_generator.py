#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

from pdb import set_trace as st

from dovebirdia.datasets.nyse_dataset import nyseDataset

dataset_params = {
    'with_val':True,
    'n_securities':1,
    'n_samples':None,
    'price_types':['open','close','high','low'],
    'standardize':False,
    'feature_range':(0,1),
    'dataset_name':'nyse_all_train_test_split_n_securities_1_n_samples_None_features_4'
}

dataset = nyseDataset(params=dataset_params).getDataset()
