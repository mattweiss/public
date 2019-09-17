import numpy as np
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
import dovebirdia.utilities.dr_functions as drfns

# define domain randomization parameters
# dr_fns is a dictionary with a single key whose value is a list.
# Each element of this list is a list defining a function: [ function name, function definition, parameters ]
# if parameters is a tuple that is the range from which the parameter is drawn

dr_params = dict()
dr_params['save_path'] = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/datasets/test.pkl'
dr_params['ds_type'] = 'test'
dr_params['x_range'] = (-1,1)
dr_params['n_trials'] = 10
dr_params['n_samples'] = 100
dr_params['n_features'] = 1
n = 10.0
dr_params['fns'] = [
    #['exponential', drfns.exponential_fn, [1.0,(0.02,0.045),-1.0]],
    #['sigmoid', drfns.sigmoid_fn, [(0.0,100.0),0.15,60.0]],
    ['taylor_poly', drfns.taylor_poly, [(-n,n),(-n,n),(-n,n),(-n,n)]],
    #['legendre_poly', drfns.legendre_poly, [1.0,(-n,n),(-n,n),(-n,n)]],
]
dr_params['noise'] = np.random.normal
dr_params['noise_params'] = {'loc':0.0, 'scale':1.0}

# create DomainRandomizationDataset object
dr_dataset = DomainRandomizationDataset(dr_params)

# generate dataset and save to disk
dr_dataset.generateDataset()
