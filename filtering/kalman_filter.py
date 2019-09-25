import tensorflow as tf
import numpy as np
#from sklearn.datasets import make_spd_matrix
from scipy import stats
from pdb import set_trace as st
from dovebirdia.filtering.base import AbstractFilter
from dovebirdia.utilities.base import saveDict

class KalmanFilter(AbstractFilter):

    def __init__(self, params=None):

        """ 
        Implement a Kalman Filter in Tensorflow
        """

        super().__init__(params)

        self._dt = self._sample_freq**-1

        # constructs matrices based on NCV, NCA, etc.
        self._F, self._Q, self._H, self._R = self._buildModel()
        
        self._x0 = tf.constant(np.zeros((self._dimensions[1]*self._n_signals,1), dtype=np.float64), dtype=tf.float64, name='x0')
        self._z0 = tf.constant(np.zeros((self._n_signals,1), dtype=np.float64), dtype=tf.float64, name='z0')
        self._P0 = tf.constant(np.eye( self._dimensions[1]*self._n_signals, dtype=np.float64), dtype=tf.float64, name='P0')
        #self._P0 = tf.constant(make_spd_matrix( self._dimensions[1]*self._n_signals ), dtype=tf.float64, name='P0')

################################################################################

    def evaluate(self, x=None, y=None, t=None, x_key='z_hat_post', save_results=True):

        assert x is not None
        assert y is not None
        assert t is not None

        test_loss = list()
        x_hat_list = list()
        sw_list = list()
        
        for X,Y in zip(x,y):
        
            filter_results = self.filter(X)

            x_hat = filter_results[x_key][:,:,0]
            
            test_loss.append(np.square(np.subtract(x_hat,Y)).mean())
            x_hat_list.append(x_hat)

            # Shapiro-Wilk test
            w, p = stats.shapiro(x-x_hat)
            sw_list.append(p)
        
        x_hat = np.asarray(x_hat_list)
        sw = np.asarray(sw_list)
        
        # save predictions
        if save_results:

            test_results_dict = {
                'x':x,
                'y':y,
                'x_hat':x_hat,
                't':t,
                'sw':sw,
                }
            

            saveDict(save_dict=test_results_dict, save_path='./results/testing_results.pkl')

        return {'test_loss':test_loss}

################################################################################

    def filter(self, inputs):

        """ 
        Apply Kalman Filter, Using Wrapper Functions
        inputs is a list.  First element is z, second (optional) element is R
        """

        try:
            
            # extract z and (possibly) R from inputs list
            z, R = inputs
            self._z = tf.convert_to_tensor(inputs[0])
            self._R = inputs[1]

        except:

            # if R is not passed set z
            z = inputs
            self._z = tf.convert_to_tensor(inputs)

        self._x_hat_pri, self._x_hat_post,\
        self._z_hat_pri, self._z_hat_post,\
        self._P_hat_pri, self._P_hat_post, self._kf_ctr = tf.scan(self._kfScan,
                                                                  self._z,
                                                                  initializer = [ self._x0, self._x0,
                                                                                  self._z0, self._z0,
                                                                                  self._P0, self._P0,
                                                                                  tf.constant(0) ], name='kfScan')

        # if z is numpy array run session
        if not isinstance(z,tf.Tensor):

            with tf.Session() as sess:

                self._x_hat_pri = sess.run(self._x_hat_pri)
                self._x_hat_post = sess.run(self._x_hat_post)
                self._z_hat_pri = sess.run(self._z_hat_pri)
                self._z_hat_post = sess.run(self._z_hat_post)
                self._P_hat_pri = sess.run(self._P_hat_pri)
                self._P_hat_post = sess.run(self._P_hat_post)
                self._z = sess.run(self._z)
            
        return { 'x_hat_pri':self._x_hat_pri, 'x_hat_post':self._x_hat_post,\
                 'z_hat_pri':self._z_hat_pri, 'z_hat_post':self._z_hat_post,\
                 'P_hat_pri':self._P_hat_pri, 'P_hat_post':self._P_hat_post,\
                 'z':self._z }
            
################################################################################

    def _kfScan(self, state, z):
        
        """ This is where the acutal Kalman Filter is implemented. """

        x_pri, x_post, z_pri, z_post, P_pri, P_post, self._kf_ctr = state

        # reset state estimate each minibatch
        x_pri, x_post, z_pri, z_post, P_pri, P_post, self._kf_ctr = tf.cond( tf.less( self._kf_ctr, self._n_samples ),
                                                                             lambda: [ x_pri, x_post, z_pri, z_post, P_pri, P_post, self._kf_ctr ],
                                                                             lambda: [ self._x0, self._x0, self._z0, self._z0, self._P0, self._P0, tf.constant(0) ])

        z = tf.expand_dims( z, axis = -1 )

        # Predict
        x_pri = tf.matmul( self._F, x_post, name='x_pri' )
        P_pri = tf.add( tf.matmul( self._F, tf.matmul( P_post, self._F, transpose_b=True ) ), self._Q, name='P_pri' )

        # assume R is scalar
        try:

            R = self._R[self._kf_ctr,:,:]

        except:

            R = self._R

        S = tf.matmul(self._H, tf.matmul(P_pri, self._H, transpose_b=True)) + R
        S_inv = tf.linalg.inv( S )

        K = tf.matmul( P_pri, tf.matmul( self._H, S_inv, transpose_a=True, name = 'KF_H-S_inv' ), name='KF_K' )

        # Update
        innov_plus = tf.subtract( z, tf.matmul( self._H, x_pri ), name='innov_plus' )
        x_post = tf.add( x_pri, tf.matmul( K, innov_plus ), name = 'x_post' )
        P_post = tf.matmul( tf.subtract( tf.eye( tf.shape( P_pri )[0], dtype=tf.float64), tf.matmul( K, self._H ) ), P_pri, name = 'P_post' )

        # map state estimates to measurement space
        z_pri  = tf.matmul(self._H, x_pri, name='z_pri', transpose_b=False)
        z_post = tf.matmul(self._H, x_post, name='z_post', transpose_b=False)
        
        return [ x_pri, x_post, z_pri, z_post, P_pri, P_post, tf.add(self._kf_ctr,1) ]

###############################################################################

    def _dictToAttributes(self,params):

        # Assign Attributes
        for key, value in params.items():

            #if isinstance(value,int) or value is None:

            setattr(self, '_' + key, value)

            # else:

            #     setattr(self, '_' + key, tf.constant(value, name=key, dtype=tf.float64) )

    def _buildModel(self):

        """
        Builds F, Q, H and R based on inputs
        """

        # F = tf.constant(np.kron(np.eye(self._n_signals), np.array([[1.0,self._dt],[0.0,1.0]], dtype=np.float64)), dtype=tf.float64, name='F')
        # Q = tf.constant(np.kron(np.eye(self._n_signals), np.array([[self._q,0.0],[0.0,self._q]], dtype=np.float64)), dtype=tf.float64, name='Q')
        # H = tf.constant(np.kron(np.eye(self._n_signals), np.array(self._h, dtype=np.float64)), dtype=tf.float64, name='H')

        # ZV
        if self._dimensions[1] == 1:

            F = np.kron(np.eye(self._n_signals), np.array([[1.0]]))
            Q = np.kron(np.eye(self._n_signals), np.array([[self._q]]))
            H = np.kron(np.eye(self._n_signals), np.array([self._h]))

        # NCV
        elif self._dimensions[1] == 2:

            F = np.kron(np.eye(self._n_signals), np.array([[1.0,self._dt],[0.0,1.0]]))
            Q = np.kron(np.eye(self._n_signals), np.array([[self._q,0.0],[0.0,self._q]]))
            H = np.kron(np.eye(self._n_signals), np.array([self._h,0.0]))

        # NCA
        elif self._dimensions[1] == 3:

            F = np.kron(np.eye(self._n_signals), np.array([[1.0,self._dt,0.5*self._dt**2],[0.0,1.0,self._dt],[0.0,0.0,1.0]]))
            Q = np.kron(np.eye(self._n_signals), np.array([[self._q,0.0,0.0],[0.0,self._q,0.0],[0.0,0.0,self._q]]))
            H = np.kron(np.eye(self._n_signals), np.array([self._h,0.0,0.0]))

        # jerk model
        elif self._dimensions[1] == 4:

            F = np.kron(np.eye(self._n_signals), np.array([[1.0, self._dt, 0.5*self._dt**2, (6**-1)*self._dt**3],
                                                             [0.0,1.0,self._dt, 0.5*self._dt**2],
                                                             [0.0,0.0,1.0,self._dt],
                                                             [0.0,0.0,0.0,1.0]]))

            Q = np.kron(np.eye(self._n_signals), np.array([[self._q,0.0,0.0,0.0],
                                                             [0.0,self._q,0.0,0.0],
                                                             [0.0,0.0,self._q,0.0],
                                                             [0.0,0.0,0.0,self._q]]))

            H = np.kron(np.eye(self._n_signals), np.array([self._h,0.0,0.0,0.0]))

                    
        # set R if parameter was passed
        try:

            R = tf.constant(np.kron(np.eye(self._n_signals), np.array(self._r, dtype=np.float64)), dtype=tf.float64, name='R')

        except:

            R = None

        return F, Q, H, R
