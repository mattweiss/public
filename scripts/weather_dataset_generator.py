#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

from pdb import set_trace as st

from dovebirdia.datasets.weather_dataset import weatherDataset

dataset_params = {
    'datadir':'/home/mlweiss/Documents/wpi/research/data/weather/historicalHourlyWeatherData/',
    'n_samples':(0,40000),
    'city':'Phoenix',
    'standardize':True,
    'features':['temperature','pressure'] #,'humidity','wind_direction','wind_speed'],
}

dataset_params['dataset_name'] = 'hourly_weather_dataset_FEATURES_{features}_CITY_{city}_SAMPLES_{samples}'.format(features=('_').join(dataset_params['features']),
                                                                                                                   city=dataset_params['city'],
                                                                                                                   samples=dataset_params['n_samples'][1])

dataset = weatherDataset(params=dataset_params).getDataset()
