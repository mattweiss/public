import numpy as np
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.datasets.flight_kinematics import FlightKinematicsDataset
import dovebirdia.utilities.dr_functions as drfns
import dovebirdia.math.distributions as distributions

# define domain randomization parameters
# dr_fns is a dictionary with a single key whose value is a list.
# Each element of this list is a list defining a function: [ function name, function definition, parameters ]
# if parameters is a tuple that is the range from which the parameter is drawn

ds_params = dict()
ds_params['save_path'] = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/dissertation/imm/eval/benchmark_gaussian_20_turn.pkl'

#######################################
trials=1000
turns = 1
samples_per_state = 100
samples = (2*turns+1)*samples_per_state
dt=0.1
#######################################

ds_params['n_trials']=trials
ds_params['n_samples']=samples
ds_params['n_turns']=turns
ds_params['dt']=dt
ds_params['r0']=(0.0,0.0)
ds_params['v0']=(100.0,0.0)
ds_params['radius_range']=(200.0,300.0)
ds_params['angle_range']=(np.pi/4,np.pi/4)
ds_params['cw']=1
ds_params['noise']=(np.random.normal,{'loc':0.0,'scale':20},1.0)
#ds_params['noise']=(np.random.standard_cauchy,{},5.0) # last entry is manual scale for cauchy

# create DomainRandomizationDataset object
dr_dataset = FlightKinematicsDataset(**ds_params)

# generate dataset and save to disk
dr_dataset.generateDataset()
