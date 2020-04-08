import sys
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_spd_matrix
from scipy import stats
from pdb import set_trace as st
from dovebirdia.filtering.base import AbstractFilter
from dovebirdia.utilities.base import saveDict

class KalmanFilter(AbstractFilter):

    def __init__(self, params=None):

        """
        Implements a Kalman Filter in Tensorflow
        """
        params['sample_freq'] = np.reciprocal(params['dt'])

        super().__init__(params)

################################################################################

    def fit(self, inputs):

        """
        Apply Kalman Filter, Using Wrapper Functions
        inputs is a list.  First element is z, second (optional) element is R
        """

        if isinstance(inputs,list):

            # extract z and (possibly) R from inputs list
            z = inputs[0]
            self._R = inputs[1]

            # ensure z is rank e
            if np.ndim(z) < 3:

                np.expand_dims(z,axis=-1)

            z = tf.convert_to_tensor(inputs[0])
            
        else:

            # if R is not passed set z
            # ensure z is rank e
            if np.ndim(inputs) < 3:

                inputs = np.expand_dims(inputs,axis=-1)

            z = tf.convert_to_tensor(inputs)

        # set x0 to initial mesurement, set all derviatives to zero
        # x0_dots = tf.zeros( shape=(self._state_dims,self._model_order), dtype=tf.float64, name='x0_dots') # n_signals x 1
        # x0 = z[0,::self._model_order+1] if self._with_z_dot else z[0]
        # self._x0 = tf.reshape(tf.concat([x0,x0_dots],axis=1),[-1, tf.shape(x0)[1]])
        # self._P0 = tf.matmul(self._x0,self._x0,transpose_b=True)
        
        self._x0 = np.zeros(((self._model_order+1)*self._state_dims,1), dtype=np.float64)
        self._P0 = np.eye((self._model_order+1)*self._state_dims, dtype=np.float64)
        
        x_hat_pri, x_hat_post, P_hat_pri, P_hat_post, self._kf_ctr = tf.scan(self._kfScan,
                                                                             z,
                                                                             initializer = [ self._x0, self._x0, self._P0, self._P0, tf.constant(0) ], name='kfScan')

        z_hat_pri  = tf.matmul(self._H, x_hat_pri, name='z_pri', transpose_b=False)
        z_hat_post = tf.matmul(self._H, x_hat_post, name='z_post', transpose_b=False)

        filter_result = {
            'x_hat_pri':x_hat_pri, 'x_hat_post':x_hat_post,
            'z_hat_pri':z_hat_pri, 'z_hat_post':z_hat_post,
            'P_hat_pri':P_hat_pri, 'P_hat_post':P_hat_post,
            'z':z,
            'R':self._R,
            }

        # if session is currently defined try will fail and tensors will be returned, otherwise evaluate tensors and return np arrays
        try:

            sess = tf.InteractiveSession()
            filter_result_np, R = sess.run([filter_result,self._R])
            sess.close()

        except:

            filter_result_np = filter_result
            # x_hat = filter_result[x_key][:,:,0]
            # R = self._R

        return filter_result_np

################################################################################

    def evaluate(self, x=None, x_key='z_hat_post', save_results=True):

        assert x is not None

        filter_result = self.fit(x)

        return filter_result[x_key][:,:,0], filter_result['R']

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

        x_post, P_post = self._update(z, x_pri, P_pri)
        
        return [ x_pri, x_post, P_pri, P_post, tf.add(self._kf_ctr,1) ]

###############################################################################

    def _predict(self,x=None,P=None):

        assert x is not None
        assert P is not None

        x_pri = tf.matmul(self._F, x, name='x_pri' )
        P_pri = tf.add( tf.matmul( self._F, tf.matmul( P, self._F, transpose_b=True ) ), self._Q, name='P_pri' )

        return x_pri, P_pri

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
            
        S = tf.matmul(self._H, tf.matmul(P, self._H, transpose_b=True)) + R
        S_inv = tf.linalg.inv(S)

        K = tf.matmul(P,tf.matmul(self._H, S_inv, transpose_a=True, name = 'KF_H-S_inv' ), name='KF_K' )

        innov_plus = tf.subtract( z, tf.matmul( self._H, x ), name='innov_plus' )

        x_post = tf.add( x, tf.matmul( K, innov_plus ), name = 'x_post' )
        P_post = tf.matmul( tf.subtract( tf.eye( tf.shape( P )[0], dtype=tf.float64), tf.matmul( K, self._H ) ), P, name = 'P_post' )

        return x, P
        
###############################################################################
    #def _buildModel(self):

        # """
        # Builds F, Q, H and R based on inputs
        # """
        # ZV
        # if self._dimensions[1] == 1:

        #     if self._f_model == 'fixed':

        #         F = np.kron(np.eye(self._n_signals), np.array([[1.0]]))

        #     elif self._f_model == 'random':

        #         F = np.kron(np.eye(self._n_signals), np.array([[1.0]]))

        #     elif self._f_model == 'learned':

        #         F = tf.get_variable(name='F',
        #                             shape=(self._n_signals*self._dimensions[1],self._n_signals*self._dimensions[1]),
        #                             initializer=tf.initializers.glorot_normal,
        #                             trainable=True,
        #                             dtype=tf.float64)

        #     Q = np.kron(np.eye(self._n_signals), np.array([[self._q]]))
        #     H = np.kron(np.eye(self._n_signals), np.array([self._h]))

        # # NCV
        # elif self._dimensions[1] == 2:

        #     st()
            
        #     if hasattr(self,'_F'):

        #         F = self._F

        #     elif self._f_model == 'learned':

        #         with tf.variable_scope(name_or_scope='KF',
        #                                regularizer=None,
        #                                reuse=tf.AUTO_REUSE):

        #             F = tf.get_variable(name='F',
        #                                 shape=(self._n_signals*self._dimensions[1],self._n_signals*self._dimensions[1]),
        #                                 initializer=self._weight_initializer,
        #                                 trainable=True,
        #                                 dtype=tf.float64)

        #     # backwards compatibility
        #     else:

        #         F = np.kron(np.eye(self._n_signals), np.array([[1.0,self._sample_freq**-1],[0.0,1.0]]))

        #     if hasattr(self,'_H'):

        #         H = self._H

        #     elif self._h_model == 'learned':

        #         with tf.variable_scope(name_or_scope='KF',
        #                                regularizer=None,
        #                                reuse=tf.AUTO_REUSE):

        #             H = tf.get_variable(name='H',
        #                                 shape=(self._n_signals,self._n_signals*self._dimensions[1]),
        #                                 initializer=self._weight_initializer,
        #                                 trainable=True,
        #                                 dtype=tf.float64)

        #     # backwards compatibility
        #     else:

        #         H = np.kron(np.eye(self._n_signals), np.array([1.0,0.0]))

        #     # Q = np.kron(np.eye(self._n_signals), np.array([[self._q,0.0],[0.0,self._q]]))
        #     Q = np.kron(np.eye(self._n_signals),np.full((self._dimensions[1],self._dimensions[1]), self._q))

        # # NCA
        # elif self._dimensions[1] == 3:

        #     if hasattr(self,'_F'):

        #         F = self._F

        #     elif self._f_model == 'learned':

        #         with tf.variable_scope(name_or_scope='KF',
        #                                regularizer=None,
        #                                reuse=tf.AUTO_REUSE):

        #             F = tf.get_variable(name='F',
        #                                 shape=(self._n_signals*self._dimensions[1],self._n_signals*self._dimensions[1]),
        #                                 initializer=self._weight_initializer,
        #                                 trainable=True,
        #                                 dtype=tf.float64)

        #     if hasattr(self,'_H'):

        #         H = self._H

        #     elif self._h_model == 'learned':

        #         with tf.variable_scope(name_or_scope='KF',
        #                                regularizer=None,
        #                                reuse=tf.AUTO_REUSE):

        #             H = tf.get_variable(name='H',
        #                                 shape=(self._n_signals,self._n_signals*self._dimensions[1]),
        #                                 initializer=self._weight_initializer,
        #                                 trainable=True,
        #                                 dtype=tf.float64)

        #     # if self._f_model == 'fixed':

        #     #     F = np.kron(np.eye(self._n_signals), np.array([[1.0,self._dt,0.5*self._dt**2],[0.0,1.0,self._dt],[0.0,0.0,1.0]]))

        #     # elif self._f_model == 'random':

        #     #     F = np.random.normal(size=(self._n_signals*self._dimensions[1],self._n_signals*self._dimensions[1]))

        #     # elif self._f_model == 'learned':

        #     #     F = tf.get_variable(name='F',
        #     #                         shape=(self._n_signals*self._dimensions[1],self._n_signals*self._dimensions[1]),
        #     #                         initializer=tf.initializers.glorot_normal,
        #     #                         trainable=True,
        #     #                         dtype=tf.float64)

        #     #F = np.kron(np.eye(self._n_signals), np.array([[1.0,self._dt,0.5*self._dt**2],[0.0,1.0,self._dt],[0.0,0.0,1.0]]))
        #     Q = np.kron(np.eye(self._n_signals), np.array([[self._q,0.0,0.0],[0.0,self._q,0.0],[0.0,0.0,self._q]]))
        #     #H = np.kron(np.eye(self._n_signals), np.array([self._h,0.0,0.0]))

        # # jerk model
        # elif self._dimensions[1] == 4:

        #     F = np.kron(np.eye(self._n_signals), np.array([[1.0, self._dt, 0.5*self._dt**2, (6**-1)*self._dt**3],
        #                                                      [0.0,1.0,self._dt, 0.5*self._dt**2],
        #                                                      [0.0,0.0,1.0,self._dt],
        #                                                      [0.0,0.0,0.0,1.0]]))

        #     Q = np.kron(np.eye(self._n_signals), np.array([[self._q,0.0,0.0,0.0],
        #                                                      [0.0,self._q,0.0,0.0],
        #                                                      [0.0,0.0,self._q,0.0],
        #                                                      [0.0,0.0,0.0,self._q]]))

        #     H = np.kron(np.eye(self._n_signals), np.array([self._h,0.0,0.0,0.0]))


        # set R if parameter was passed
        # try:

        #     R = tf.constant(np.kron(np.eye(self._n_signals), np.array(self._r, dtype=np.float64)), dtype=tf.float64, name='R')

        # except:

        #     R = None

        # R is treated differenly since it may be learned by AEKF

        #return params['F'], params['Q'], params['H'], params['R'] # params['R'] is None if R is learned, i.e. using AEKF
