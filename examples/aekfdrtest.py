import numpy as np
import numpy.polynomial.polynomial as poly
import numpy.polynomial.legendre as leg_poly
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.autoencoder import AutoencoderKalmanFilter
import dovebirdia.utilities.dr_functions as drfns

# define domain randomization parameters
# dr_fns is a dictionary with a single key whose value is a list.
# Each element of this list is a list defining a function: [ function name, function definition, parameters ]
# if parameters is a tuple that is the range from which the parameter is drawn

dr_params = dict()
dr_params['ds_type'] = 'train'
dr_params['x_range'] = (-1,1)
dr_params['n_trials'] = 1
dr_params['n_samples'] = 100
dr_params['n_features'] = 1
n = 1.0
dr_params['fns'] = [
    #['exponential', drfns.exponential_fn, [1.0,(0.02,0.045),-1.0]],
    #['sigmoid', drfns.sigmoid_fn, [(0.0,100.0),0.15,60.0]],
    ['taylor_poly', drfns.taylor_poly, [(-n,n),(-n,n),(-n,n),(-n,n)]],
    #['legendre_poly', drfns.legendre_poly, [1.0,(-n,n),(-n,n),(-n,n)]],
]
dr_params['noise'] = np.random.normal
dr_params['noise_params'] = {'loc':0.0, 'scale':0.1}

# parameters dictionary
model_params = dict()

# output params
model_params['model_name'] = 'aedrtest'
model_params['results_dir'] = './saved_weights/'

# network params
model_params['input_dim'] = 1
model_params['output_dim'] = 1
model_params['hidden_dims'] = [ 256,64,16 ]
model_params['output_activation'] = None
model_params['activation'] = tf.nn.relu
model_params['use_bias'] = True
model_params['kernel_initializer'] = 'glorot_uniform'
model_params['bias_initializer'] = 'zeros'
model_params['kernel_regularizer'] = None
model_params['bias_regularizer'] = None
model_params['activity_regularizer'] = None
model_params['kernel_constraint'] = None
model_params['bias_constraint'] = None

# loss
model_params['loss'] = tf.losses.mean_squared_error
model_params['history_size'] = 100

# training
model_params['epochs'] = 1000
model_params['mbsize'] = 100
model_params['optimizer'] = tf.train.AdamOptimizer
model_params['learning_rate'] = 1e-3
model_params['test_size'] = 100

# KF params
kf_params = dict()
kf_params['dimensions'] = (1,2)
#kf_params['model'] = 'ncv'
kf_params['n_signals'] = model_params['hidden_dims'][-1]
kf_params['n_samples'] = dr_params['n_samples']
kf_params['sample_freq'] = (dr_params['x_range'][1]-dr_params['x_range'][0]) * dr_params['n_samples']
kf_params['h'] = [1.0,0.0]
kf_params['q'] = 1e-4

# Network
nn = AutoencoderKalmanFilter(model_params, kf_params)
print(nn.__class__)
nn.getModelSummary()
nn.fitDomainRandomization(dr_params, save_weights=False)
