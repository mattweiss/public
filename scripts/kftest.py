import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.filtering.kalman_filter import KalmanFilter

kf_params = dict()
kf_params['dimensions'] = (1,2)
#kf_params['model'] = 'ncv'
kf_params['n_signals'] = 1
kf_params['n_samples'] = 100
kf_params['sample_freq'] = 1.0
kf_params['r'] = 100.0
#R = np.kron(np.eye( kf_params['n_signals']), np.array(kf_params['r'], dtype=np.float64))
kf_params['h'] = [1.0,0.0]
kf_params['q'] = 1e-2
# kf_params['dt'] = kf_params['sample_freq']**-1
# kf_params['F'] = np.kron(np.eye(n_signals), np.array([[1.0,kf_params['dt']],[0.0,1.0]], dtype=np.float64))
# kf_params['Q'] = np.kron(np.eye(n_signals), np.array(q, dtype=np.float64))
# kf_params['H'] = np.kron(np.eye(n_signals), np.array(h, dtype=np.float64))
# kf_params['R'] = np.kron(np.eye(n_signals), np.array(r, dtype=np.float64))
# kf_params['x0'] = np.zeros((kf_params['dimensions'][1]*kf_params['n_signals'],1), dtype=np.float64)
# kf_params['z0'] = np.zeros((kf_params['n_signals'],1), dtype=np.float64)
# kf_params['P0'] = np.eye( kf_params['dimensions'][1]*kf_params['n_signals'], dtype=np.float64 )

for k,v in kf_params.items():

    try:

        print(k,v.shape)

    except:

        print(k,v)

x = np.linspace(0,10,100)
z = np.expand_dims((lambda x: np.exp(0.4*x))(x) + np.random.normal(scale=5.0, size=x.shape).astype(np.float64), axis=-1)

kf_results = KalmanFilter(kf_params).filter(z)
plt.plot(z, label='z')
plt.plot(kf_results['x_hat_post'][:,0], label='x post')
plt.plot(kf_results['x_hat_pri'][:,0], label='x pri')
plt.plot(kf_results['x_hat_post'][:,1], label='x-dot post')
plt.legend()
plt.grid()
plt.show()
