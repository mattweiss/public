import tensorflow as tf
import numpy as np
from pdb import set_trace as st
from dovebirdia.filtering.base import AbstractFilter

class KalmanFilter(AbstractFilter):

    def __init__(self, params=None):

        """ 
        Implement a Kalman Filter in Tensorflow
        """

        super().__init__(params)

        # self._z = tf.placeholder(dtype=tf.float64, shape=(None,self._n_signals), name='z')
   
################################################################################

    def filter(self, inputs):

        """ 
        Apply Kalman Filter, Using Wrapper Functions
        inputs is a list.  First element is z, second (optional) element is R
        """

        # extract z and (possibly) R from inputs list
        self._z = tf.convert_to_tensor(inputs[0])

        try:

            self._R = inputs[1]

        except:

            pass
        
        self._x_hat_pri, self._x_hat_post,\
        self._z_hat_pri, self._z_hat_post,\
        self._P_hat_pri, self._P_hat_post, self._kf_ctr = tf.scan(self._kfScan,
                                                                  self._z,
                                                                  initializer = [ self._x0, self._x0,
                                                                                  self._z0, self._z0,
                                                                                  self._P0, self._P0,
                                                                                  tf.constant(0) ], name='kfScan')

        #cond_bool = tf.cond(tf.cast(tf.is_tensor(z),tf.bool), lambda: True, lambda: self._runSess())
        
        # if z is numpy array run session
        if not isinstance(inputs[0],tf.Tensor):

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
        P_post = tf.matmul( tf.subtract( tf.eye( tf.shape( P_pri )[0], dtype=tf.float64 ), tf.matmul( K, self._H ) ), P_pri, name = 'P_post' )

        # map state estimates to measurement space
        z_pri  = tf.matmul(self._H, x_pri, name='z_pri', transpose_b=False)
        z_post = tf.matmul(self._H, x_post, name='z_post', transpose_b=False)
        
        return [ x_pri, x_post, z_pri, z_post, P_pri, P_post, tf.add(self._kf_ctr,1) ]

###############################################################################

    def _dictToAttributes(self,params):

        # Assign Attributes
        for key, value in params.items():

            if isinstance(value,int) or value is None:

                setattr(self, '_' + key, value)

            else:

                setattr(self, '_' + key, tf.constant(value, name=key, dtype=tf.float64) )
