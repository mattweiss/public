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
dr_params['save_path'] = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/evaluation/sdm/FUNC_sine_NOISE_gaussian_LOC_0_SCALE_5_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl'
dr_params['ds_type'] = 'test'
dr_params['x_range'] = (0,100)
dr_params['n_trials'] = 100
dr_params['n_baseline_samples'] = 10
dr_params['n_samples'] = 100
dr_params['n_features'] = 1

#n = 10.0
dr_params['fns'] = (
    #['exponential', drfns.exponential, [1.0,(0.02,0.045),-1.0]],
    #['sigmoid', drfns.sigmoid, [(0.0,100.0),0.15,60.0]],
    ['sine', drfns.sine, [(0.0,100.0),(0.04,0.1)]],
    #['taylor_poly', drfns.taylor_poly, [(-n,n),(-n,n),(-n,n),(-n,n)]],
    #['legendre_poly', drfns.legendre_poly, [(-n,n),(-n,n),(-n,n),(-n,n)]],
)

dr_params['noise'] = (
    ['gaussian', np.random.normal, {'loc':0.0, 'scale':5.0}],
    #['bimodal', distributions.bimodal, {'loc1':3.0, 'scale1':1.0, 'loc2':-3.0, 'scale2':1.0}],
    #['cauchy', np.random.standard_cauchy, {}],
    #['stable', distributions.stable, {'alpha':(1.0,2.0)}],
    #['stable', distributions.stable, {'alpha':1.0}],
    # ['stable', distributions.stable, {'alpha':1.5}],
    #['stable', distributions.stable, {'alpha':2.0}],
)

# create DomainRandomizationDataset object
dr_dataset = DomainRandomizationDataset(dr_params)

# generate dataset and save to disk
dr_dataset.generateDataset()
