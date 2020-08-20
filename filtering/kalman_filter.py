import sys
import tensorflow as tf
tf_float_prec = tf.float64
import numpy as np
np_float_prec = np.float64
import tensorflow_probability as tfp
from sklearn.datasets import make_spd_matrix
from scipy import stats
from pdb import set_trace as st
from dovebirdia.filtering.base import AbstractFilter
from dovebirdia.utilities.base import saveDict
from dovebirdia.math.linalg import is_invertible, pos_diag

class KalmanFilter(AbstractFilter):

    def __init__(self, params=None):

        """
        Implements a Kalman Filter in Tensorflow
        """

        params['sample_freq'] = np.reciprocal(params['dt'])

        super().__init__(params)
        
        self._x0 = np.zeros(((self._model_order+1)*self._state_dims,1), dtype=np_float_prec)
        self._P0 = np.eye((self._model_order+1)*self._state_dims, dtype=np_float_prec)
        
################################################################################

    def fit(self, inputs):

        """
        Apply Kalman Filter, Using Wrapper Functions
        inputs is a list.  First element is z, second (optional) element is R
        """

        z = self._process_inputs(inputs)

        x_hat_pri, x_hat_post, P_pri, P_post, self._kf_ctr = tf.scan(self._kfScan,
                                                                     z,
                                                                     initializer = [ self._x0,
                                                                                     self._x0,
                                                                                     self._P0,
                                                                                     self._P0, tf.constant(0) ],
                                                                     name='kfScan')

        filter_results = self._process_results(x_hat_pri, x_hat_post, P_pri, P_post, z)

        return filter_results

################################################################################

    def _kfScan(self, state, z):

        """ This is where the Kalman Filter is implemented. """

        _, x_post, _, P_post, self._kf_ctr = state

        ##########
        # Predict
        ##########
        
        x_pri, P_pri = self._predict(x_post,P_post)

        #########
        # Update
        #########

        x_post, P_post, _ = self._update(z, x_pri, P_pri)

        return [ x_pri, x_post, P_pri, P_post, tf.add(self._kf_ctr,1) ]

###############################################################################

    def _predict(self,x=None,P=None):

        assert x is not None
        assert P is not None

        x_pri = tf.matmul(self._F,x,name='x_pri')
        P_pri = tf.add(tf.matmul(self._F,tf.matmul(P,self._F,transpose_b=True)),self._Q,name='P_pri')
        
        return x_pri, P_pri

###############################################################################

    def _update(self,z,x,P):

        assert z is not None
        assert x is not None
        assert P is not None

        # indexed R
        try:

            R = self._R[self._kf_ctr]

        # Fixed R
        except:

            R = self._R

        S = tf.matmul(self._H,tf.matmul(P,self._H,transpose_b=True)) + R #+ tf.cast(1e-1*tf.eye(R.shape[0]),dtype=tf_float_prec)

        S_inv = tf.linalg.inv(S)

        K = tf.matmul(P,tf.matmul(self._H,S_inv,transpose_a=True,name='KF_H-S_inv'),name='KF_K')

        y = tf.subtract(z,tf.matmul(self._H,x),name='innov_plus')

        x_post = tf.add(x,tf.matmul(K,y),name='x_post')
        P_post = (tf.eye(tf.shape(P)[0],dtype=tf_float_prec)-K@self._H)@P

        # compute likelihood
        likelihood = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(y.get_shape()[0],dtype=tf_float_prec),
                                                                        covariance_matrix=S).prob(tf.transpose(y))[0]

        likelihood = likelihood + tf.cast(1e-8,dtype=tf_float_prec)
        
        return x_post, P_post, likelihood

################################################################################

    def _process_inputs(self,inputs):

        # if learning R, z and R will be passed
        if isinstance(inputs,list):

            # extract z and (possibly) R from inputs list
            z = inputs[0]
            self._R = inputs[1]

            # ensure z is rank e
            if np.ndim(z) < 3:

                np.expand_dims(z,axis=-1)

            z = tf.convert_to_tensor(z)

        # fixed R or sample R
        else:

            # if R is not passed set z
            # ensure z is rank 3
            if np.ndim(inputs) < 3:

                z = np.expand_dims(inputs,axis=-1)
                
            # if self._R is non use sample covariance
            if self._R is None:

                z_hat = np.squeeze(z) - np.mean(np.squeeze(z),axis=0)
                self._R = z_hat.T@z_hat / (z_hat.shape[0]-1)
                
        return tf.convert_to_tensor(z)

################################################################################

    def _process_results(self,x_hat_pri, x_hat_post, P_pri, P_post, z):

        z_hat_pri  = tf.matmul(self._H, x_hat_pri, name='z_pri', transpose_b=False)
        z_hat_post = tf.matmul(self._H, x_hat_post, name='z_post', transpose_b=False)
        HPHT_pri = self._H@P_pri@tf.transpose(self._H)
        HPHT_post = self._H@P_post@tf.transpose(self._H)
        
        filter_result = {
            'x_hat_pri':x_hat_pri,
            'x_hat_post':x_hat_post,
            'z_hat_pri':z_hat_pri,
            'z_hat_post':z_hat_post,
            'P_pri':P_pri,
            'P_post':P_post,
            'HPHT_pri':HPHT_pri,
            'HPHT_post':HPHT_post,
            'z':z,
            'R':tf.convert_to_tensor(self._R),
            }

        # if session is currently defined try will fail and tensors will be returned, otherwise evaluate tensors and return np arrays
        try:

            sess = tf.InteractiveSession()
            filter_result = sess.run(filter_result)
            sess.close()
            
        except Exception as e:

            pass

        
        return filter_result

################################################################################

    def evaluate(self, x=None, x_key='z_hat_post', save_results=True):

        assert x is not None

        filter_result = self.fit(x)

        return filter_result[x_key][:,:,0], filter_result['R']
    
