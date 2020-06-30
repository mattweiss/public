import numpy as np
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
import dovebirdia.utilities.dr_functions as drfns
import dovebirdia.math.distributions as distributions

# define domain randomization parameters
# dr_fns is a dictionary with a single key whose value is a list.
# Each element of this list is a list defining a function: [ function name, function definition, parameters ]
# if parameters is a tuple that is the range from which the parameter is drawn

ds_params = dict()
ds_params['save_path'] = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/nyse/eval/benchmark_legendre_cauchy_R1_1k.pkl'
ds_params['ds_type'] = 'test'
ds_params['x_range'] = (-1,1)
ds_params['n_trials'] = 10
ds_params['n_baseline_samples'] = 0
ds_params['n_samples'] = 350

# set dt here based on x range and mb size, for use in scaling noise
dt = 1.0 #(ds_params['x_range'][1]-ds_params['x_range'][0])/ds_params['n_samples']

ds_params['n_features'] = 1
ds_params['n_noise_features'] = ds_params['n_features']
ds_params['standardize'] = False
ds_params['feature_range'] = None
ds_params['baseline_shift'] = None
ds_params['param_range'] = 0.5
ds_params['max_N'] = 35
ds_params['min_N'] = 20
ds_params['fns'] = (
    #['zeros', drfns.zeros, []],
    #['exponential', drfns.exponential, [1.0,(0.02,0.045),-1.0]],
    #['sigmoid', drfns.sigmoid, [(0.0,100.0),0.15,60.0]],
    #['sine', drfns.sine, [(0.0,10.0),(0.04,0.1)]],
    #['taylor_poly', drfns.taylor_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(ds_params['max_N']+1)],
    ['legendre_poly', drfns.legendre_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(ds_params['max_N']+1)],
    #['trig_poly', drfns.trig_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(2*ds_params['max_N']+1)],
)

ds_params['noise'] = (
    #[None, None, None],

    # ['gaussian', np.random.multivariate_normal, {'mean':np.zeros(ds_params['n_features']),
    #                                              'cov':0.2*np.eye(ds_params['n_features'])}],

    # ['bimodal', distributions.bimodal, {'mean1':np.full(ds_params['n_features'],0.5),
    #                                     'cov1':0.2*np.eye(ds_params['n_features']),
    #                                     'mean2':np.full(ds_params['n_features'],-0.5),
    #                                     'cov2':0.2*np.eye(ds_params['n_features'])}],

    ['cauchy', np.random.standard_cauchy, {}],

    # ['stable', distributions.stable, {'alpha':(2.0,1.0),'scale':1.0}], # alpha = 2 Gaussian, alpha = 1 Cauchy
)

# create DomainRandomizationDataset object
dr_dataset = DomainRandomizationDataset(ds_params)

# generate dataset and save to disk
dr_dataset.generateDataset()
