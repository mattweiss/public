import sys
import tensorflow as tf
import numpy as np
np_float_prec = np.float64
from math import log
from pdb import set_trace as st
from dovebirdia.filtering.kalman_filter import KalmanFilter

class InteractingMultipleModel(KalmanFilter):

    def __init__(self, params=None):

        """
        Implements a Interacting Mixture Model Kalman Filter, derived from KalmanFilter class
        """

        super().__init__(params)

        # normalize mu, should be normalized in config file, this is a sanity check
        self._mu /= self._mu.sum()
        
        # model list
        self._model_list = list(self._models.values())

        # number of models
        self._n_models = len(self._model_list)
        
        # initialize mixing probabilities
        self._mix_prob = np.zeros((self._n_models,self._n_models))

################################################################################

    def fit(self, inputs):

        """
        Apply Kalman Filter, Using Wrapper Functions
        inputs is a list.  First element is z, second (optional) element is R
        """

        z = super()._process_inputs(inputs)

        _, _, x_hat_pri, x_hat_post, P_pri, P_post, self._kf_ctr = tf.scan(self._kfScan,
                                                                           z,
                                                                           initializer = [ [self._x0]*self._n_models, [self._P0]*self._n_models,
                                                                                           self._x0, self._x0, self._P0, self._P0, tf.constant(0) ], name='kfScan')

        filter_results = super()._process_results(x_hat_pri, x_hat_post, P_pri, P_post,z)

        return filter_results
    
################################################################################

    def _kfScan(self, state, z):

        """ This is where the Kalman Filter is implemented. """

        x_post, P_post, _ , _ , _ , _ , self._kf_ctr = state

        # Hold accumlated a priori and a posteriori x and P along with likelihood of residual
        self._x_pri = list()
        self._P_pri = list()
        self._x_post = list()
        self._P_post = list()
        self._lambda = list()
        
        # loop over models, computing a priori estimate for each 
        for model_index, model in enumerate(self._model_list): 

            ##########
            # predict
            ##########
            x_pri, P_pri = self._predict(x_post[model_index],P_post[model_index],model)

            self._x_pri.append(x_pri)
            self._P_pri.append(P_pri)
            
        # compute mixing probabilities
        self._compute_mixing_probabilities()

        # compute mixed initial estimate and covariance
        self._compute_mixed_state_and_covariance()

        # compute mixed a priori estimate and covaraince
        # this does not affect the iteration here and only used for post-processing
        x_pri_out, P_pri_out = self._combined_estimate_and_covariance(self._x_pri,self._P_pri) 
        
        # loop over models, computing a posteriori estimate for each using mixed initial conditions
        for model_index, model in enumerate(self._model_list): 

            ##########
            # update
            ##########
            x_post, P_post, Lambda = self._update(z,
                                                  self._x_pri_mixed[model_index],
                                                  self._P_pri_mixed[model_index])

            self._x_post.append(x_post)
            self._P_post.append(P_post)
            self._lambda.append(Lambda)
            
        # Mode probablity update
        self._mode_probability_update()
        
        # estimate and covaraince combination
        x_post_out, P_post_out = self._combined_estimate_and_covariance(self._x_post,self._P_post)

        return [ self._x_post, self._P_post, # input to next iteration of IMM
                 x_pri_out, x_post_out, P_pri_out, P_post_out, # returned as final output
                 tf.add(self._kf_ctr,1) ]
    
################################################################################

    def _predict(self,x,P,model):
        
        assert x is not None
        assert P is not None
        assert model is not None

        F, Q = model
                
        x_pri = tf.matmul(F,x,name='x_pri')
        P_pri = tf.add(tf.matmul(F,tf.matmul(P,F,transpose_b=True)),Q,name='P_pri' )

        return x_pri, P_pri

################################################################################

    def _update(self,z,x,P):

        x_post, P_post, Lambda = super()._update(z,x,P)
        
        return x_post, P_post, Lambda       

################################################################################

    def _compute_mixing_probabilities(self):

        """
        Estimation with Applications to Tracking and Navigation, pg. 455
        Step 1 - Calculation of Mixing Probabilities
        """
        
        # comput c_bar
        self._cbar = self._p @ self._mu

        # compute mixing parameters
        for i in range(self._n_models):

            for j in range(self._n_models):

                self._mix_prob[i][j] = (self._p[i][j] * self._mu[i]) / self._cbar[j]

################################################################################

    def _compute_mixed_state_and_covariance(self):

        """
        Estimation with Applications to Tracking and Navigation, pg. 455-456
        Step 2 - Mixing
        """

        # mixed initial conditions and covariance lists
        self._x_pri_mixed = list()
        self._P_pri_mixed = list()

        # compute mixed initial conditions
        for j in range(self._n_models):

            # temporary sum variable
            mixed_x = np.zeros(self._x0.shape).astype(np_float_prec)

            for i in range(self._n_models):

                mixed_x = tf.add(mixed_x,tf.multiply(self._x_pri[i],self._mix_prob[i][j]))
                
            self._x_pri_mixed.append(mixed_x)

        # compute mixed initial conditions covariance
        for j in range(self._n_models):

            # temporary sum variable
            mixed_P = np.zeros(self._P0.shape).astype(np_float_prec)

            for i in range(self._n_models):

                # difference between i-th model's estimate and j-th mixed initial condition
                x_diff = self._x_pri[i] - self._x_pri_mixed[j]

                mixed_P = tf.add(mixed_P,self._mix_prob[i][j] * (self._P_pri[i] + tf.matmul(x_diff,x_diff,transpose_b=True)))

            self._P_pri_mixed.append(mixed_P)

################################################################################

    def _mode_probability_update(self):

        """
        Estimation with Applications to Tracking and Navigation, pg. 456
        Step 4 - Mode Probability Update
        """

        self._mu = tf.multiply(tf.squeeze(self._cbar),tf.stack(self._lambda))
        self._mu /= tf.reduce_sum(self._mu)

################################################################################

    def _combined_estimate_and_covariance(self,x,P):

        """
        Estimation with Applications to Tracking and Navigation, pg. 457
        Step 5 - Estimate and covariance combination
        """

        x_post = np.zeros(self._x0.shape).astype(np_float_prec)
        P_post = np.zeros(self._P0.shape).astype(np_float_prec)
        
        # final state estimate
        for j in range(self._n_models):

            x_post = tf.add(x_post,tf.multiply(x[j],self._mu[j]))
        
        # final state estimate covaraince
        for j in range(self._n_models):

            x_diff = x - x_post
            P_post_j = self._mu[j] * (P[j] + tf.matmul(x_post,x_post,transpose_b=True))
            P_post = tf.add(P_post,P_post_j)
            
        return x_post, P_post

