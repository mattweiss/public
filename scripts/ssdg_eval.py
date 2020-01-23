#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

from dovebirdia.synthetic_data.homotopy_synthetic_sensor_data_generator import HomotopySyntheticSensorDataGenerator
from pdb import set_trace as st

# synthetic sensor data object
dataset_dir = '/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark/split/01_23_19/training/'
# dataset_dir = '/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark_3/split/07_12_19/training/'
#dataset_dir = '/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark/split/06_21_19-1204151148/training/'

# sensor dictionary.  -1 value is a placeholder for actual index of sensor in dataset.  This index is set in the generate_samples() method
# 01-23-19
sensors = {
    'PCL 1':-1,
    'ETCL 5':-1,
    'PECH 4':-1,
    'PEVA 4':-1,
    'PVPGT 3':-1,
    'NAF 4':-1,
    'PVA 2':-1,
    'PVPH 3':-1,
    'PVPMM 3':-1,
    '50 FLEX-NAF':-1,
    'PVA 4':-1,
    'PVPH 5':-1,
    'PVPMM 4':-1,
    'old PEVA 1':-1,
    'PCL 4':-1,
    'ETCL 1':-1,
    'PECH 5':-1,
    'PEVA 2':-1,
    'PVPGT 1':-1,
    'NAF 5':-1,
}

#02-05-19
# sensors = {

#     'PCL 1':-1,
#     'ETCL 5':-1,
#     'PECH 1':-1,
#     'PEVA 4':-1,
#     'PVPGT 3':-1,
#     'NAF 4':-1,
#     'PVA 2':-1,
#     'PVPH 3':-1,
#     'PVPMM 1':-1,
#     'PVPGT 4':-1,
#     'PVA 4':-1,
#     'PVPH 5':-1,
#     'PVPMM 4':-1,
#     'old PEVA 1':-1,
#     'PCL 5':-1,
#     'ETCL 1':-1,
#     'PECH 5':-1,
#     'PEVA 2':-1,
#     'PVPGT 1':-1,
#     'NAF 5':-1,
# }

# 06-21-19, 07-21-19
# sensors = {
#     'PCL 2':-1,
#     'PVPH 1':-1,
#     'NAF 2':-1,
#     'PVPGT 1':-1,
#     'old NAF 4':-1,
#     'old PVA 2':-1,
#     'ETCL 3':-1,
#     'NAF 4':-1,
#     'PVPMM 4':-1,
#     'PEVA 4':-1,
#     'PEVA 3':-1,
#     'PVPMM 3':-1,
#     'PECH 2':-1,
#     'ETCL 2':-1,
#     'PCL SPIN':-1,
#     'old PVA 4':-1,
#     'PVPGT 3':-1,
#     'PECH 3':-1,
#     'PVPH 2':-1,
#     'PCL 4':-1,
# }

ssdg = HomotopySyntheticSensorDataGenerator(dataset_dir = dataset_dir,
                                            trials = None,
                                            sensors = sensors,
                                            labels = None,
                                            n_synthetic_sensors_per_label = 10,
                                            use_baseline_std = True,
                                            save_plots = False)

# read pickle files
ssdg.load_data()

# generate samples
ssdg.generate_samples()
