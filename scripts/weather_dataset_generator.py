#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

from pdb import set_trace as st

from dovebirdia.datasets.weather_dataset import weatherDataset

dataset_params = {
    'n_samples':1500,
    'with_val':True,
    'support':(-1,1),
    'feature_range':(0,1),
    # 'split_len':45000, # Each city's temp has over 45000 samples.  These are split into subsamples of length split_len
    'features':['temperature','pressure','humidity','wind_direction','wind_speed'],
    'dataset_name':'weather_all_train_test_split',
    'cities':[
                # 'Vancouver',
                # 'Portland',
                # 'San Francisco',
                # 'Seattle',
                # 'Los Angeles',
                # 'San Diego',
                # 'Las Vegas',
                # 'Phoenix',
                # 'Albuquerque',
                # 'Denver',
                # 'San Antonio',
                # 'Dallas',
                # 'Houston',
                # 'Kansas City',
                # 'Minneapolis',
                # 'Saint Louis',
                # 'Chicago',
                # 'Nashville',
                # 'Indianapolis',
                # 'Atlanta',
                # 'Detroit',
                # 'Jacksonville',
                # 'Charlotte',
                # 'Miami',
                # 'Pittsburgh',
                # 'Toronto',
                # 'Philadelphia',
                # 'New York',
                # 'Montreal',
                'Boston',
                # 'Beersheba',
                # 'Tel Aviv District',
                # 'Eilat',
                # 'Haifa',
                # 'Nahariyya',
                # 'Jerusalem',
    ],
}

dataset = weatherDataset(params=dataset_params).getDataset()
