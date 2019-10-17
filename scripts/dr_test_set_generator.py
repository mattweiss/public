import numpy as np
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
import dovebirdia.utilities.dr_functions as drfns
import dovebirdia.stats.distributions as distributions

# define domain randomization parameters
# dr_fns is a dictionary with a single key whose value is a list.
# Each element of this list is a list defining a function: [ function name, function definition, parameters ]
# if parameters is a tuple that is the range from which the parameter is drawn

dr_params = dict()
dr_params['save_path'] = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/evaluation/icassp/FUNC_taylor_NOISE_bimodal_LOC_0-25_SCALE_0-2_TRIALS_1000_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl'
dr_params['ds_type'] = 'test'
dr_params['x_range'] = (-1,1)
dr_params['n_trials'] = 1000
dr_params['n_baseline_samples'] = 0
dr_params['n_samples'] = 100
dr_params['n_features'] = 1

param_range = 1.0
N=3
dr_params['fns'] = (
    #['exponential', drfns.exponential, [1.0,(0.02,0.045),-1.0]],
    #['sigmoid', drfns.sigmoid, [(0.0,100.0),0.15,60.0]],
    #['sine', drfns.sine, [(0.0,100.0),(0.04,0.1)]],
    ['taylor_poly', drfns.taylor_poly, [(-param_range,param_range)]*(N+1)],
    #['legendre_poly', drfns.legendre_poly, [(-param_range,param_range)]*(N+1)],
    #['trig_poly', drfns.trig_poly, [(-param_range,param_range)]*(2*N+1)+[2.0]],
)

dr_params['noise'] = (
    #['gaussian', np.random.normal, {'loc':0.0, 'scale':0.2}],
    ['bimodal', distributions.bimodal, {'loc1':0.25, 'scale1':0.2, 'loc2':-0.25, 'scale2':0.2}],
    #['cauchy', np.random.standard_cauchy, {}],
    #['stable', distributions.stable, {'alpha':(1.0,2.0),'scale':0.2}],
)

# create DomainRandomizationDataset object
dr_dataset = DomainRandomizationDataset(dr_params)

# generate dataset and save to disk
dr_dataset.generateDataset()
