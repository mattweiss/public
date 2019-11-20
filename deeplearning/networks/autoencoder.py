import numpy as np
import tensorflow as tf
from scipy import stats
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.layers.base import Dense
from tensorflow.python.ops.distributions.special_math import ndtri as tf_ndtri

try:
    
    from orthnet import Legendre, Chebyshev

except:

    pass
    
from sklearn.datasets import make_spd_matrix

from dovebirdia.filtering.kalman_filter import KalmanFilter
#from kalmanfilter.kalmanfiltertf import KalmanFilterTF as KalmanFilter

from dovebirdia.utilities.base import dictToAttributes, saveDict

class Autoencoder(FeedForwardNetwork):

    """
    Autoencoder Class
    """
    
    def __init__(self, params=None):

        assert isinstance(params,dict)
        
        super().__init__(params=params)
        
    ##################
    # Public Methods #
    ##################
    
    ###################
    # Private Methods #
    ###################
    
    def _buildNetwork(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')

        # encoder and decoder
        self._encoder = self._buildEncoder(input=self._X)
        self._decoder = self._buildDecoder(input=self._encoder)
        
        # output layer
        self._y_hat = self._buildOutput(input=self._decoder)

    def _buildEncoder(self, input=None):

        assert input is not None
        
        return Dense(self._hidden_layer_dict).build(input, self._hidden_dims[:-1], scope='encoder')

    def _buildDecoder(self, input=None):

        assert input is not None
        
        return Dense(self._hidden_layer_dict).build(input, self._hidden_dims[::-1][1:], scope='decoder')

    def _buildOutput(self, input=None):

        assert input is not None
         
        return Dense(self._affine_layer_dict).build(input, [self._output_dim], scope='output')
        
class AutoencoderKalmanFilter(Autoencoder):

    """
    Autoencoder-KalmanFilter Class
    """
    
    def __init__(self, params=None, kf_params=None):

        # instantiate Kalman Filter before parent constructor as
        # the parent calls _buildNetwork()
        self._kalman_filter = KalmanFilter(params=kf_params)
        
        super().__init__(params=params)

    ##################
   # Public Methods #
    ##################
 
    def evaluate(self, x=None, y=None, t=None, save_results=True):

        assert x is not None
        assert y is not None
        assert t is not None

        # model predictions
        x_hat_list = list()
        z_list = list()
        z_hat_post_list = list()
        R_list = list()

        with tf.Session() as sess:

            # backwards compatibility
            try:

                model_results_path = './results/trained_model.ckpt'
                tf.train.Saver().restore(sess, model_results_path)

            except:

                model_results_path = './results/tensorflow_model.ckpt'
                tf.train.Saver().restore(sess, model_results_path)

            for trial, (X,Y) in enumerate(zip(x,y)):

                test_feed_dict = {self._X:X, self._y:Y}

                # OKF
                if hasattr(self, '_support'):

                    support = np.linspace(-1,1,100).reshape(1,-1)
                    test_feed_dict[self._support] = support
                
                test_loss, x_hat, z, z_hat_post, R = sess.run([self._loss_op,self._y_hat,self._z,self._z_hat_post,self._R], feed_dict=test_feed_dict)
                self._history['test_loss'].append(test_loss)
                x_hat_list.append(x_hat)
                z_list.append(z)
                z_hat_post_list.append(z_hat_post)
                R_list.append(R)

        x_hat = np.asarray(x_hat_list)
        z = np.asarray(z_list)
        z_hat_post = np.asarray(z_hat_post_list)
        R = np.asarray(R_list)
        
        # save predictions
        if save_results is not None:

            test_results_dict = {
                'x':x,
                'y':y,
                'x_hat':x_hat,
                't':t,
                'z':z,
                'z_hat_post':z_hat_post,
                'R':R
                }
            
            saveDict(save_dict=test_results_dict, save_path='./results/' + save_results + '.pkl')
            
        return self._history
    
    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):
        
        # weight regularizer
        try:

            weight_regularizer = self._weight_regularizer(self._weight_regularizer_scale)

        except:

            weight_regularizer = self._weight_regularizer
            
        self._setPlaceholders()

        self._encoder = self._encoderLayer(self._X)

        self._z, self._R = self._preKalmanFilterAffineLayer(self._encoder)

        self._z_hat_pri, self._z_hat_post = self._kalmanFiterLayer([self._z, self._R])

        #self._post_kf_affine = self._postKalmanFilterAffineLayer(tf.squeeze(self._z_hat_pri,axis=-1))
        self._post_kf_affine = self._postKalmanFilterAffineLayer(self._z_hat_pri)
        
        self._decoder = self._decoderLayer(self._post_kf_affine)
        
        self._y_hat = self._outputLayer(self._decoder)
        
    def _setPlaceholders(self):
        
        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')
  
    def _encoderLayer(self, input=None):

        assert input is not None
        
        return Dense(name='encoder',
                     weight_initializer=self._weight_initializer,
                     weight_regularizer=None,
                     bias_initializer=self._bias_initializer,
                     activation=self._activation).build(input, self._hidden_dims[:-1])

    def _preKalmanFilterAffineLayer(self, input=None):

        assert input is not None
        
        z = Dense(name='z',
                        weight_initializer=self._weight_initializer,
                        weight_regularizer=None,
                        bias_initializer=self._bias_initializer,
                        activation=None).build(input, [self._hidden_dims[-1]])

        # backwards compatibility
        try:
        
            # learned noise covariance
            if self._R_model == 'learned':

                # learn L, which is vector from which SPD matrix R is formed 
                self._L_dims = np.sum(np.arange(1,self._hidden_dims[-1]+1))
                self._L = Dense(name='L',
                                weight_initializer=self._weight_initializer,
                                weight_regularizer=None,
                                bias_initializer=self._bias_initializer,
                                activation=None).build(input, [self._L_dims])

                R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

            elif self._R_model == 'identity':

                R = tf.eye(self._hidden_dims[-1], batch_shape=[tf.shape(self._z)[0]], dtype=tf.float64)
                
        except:

            # learn L, which is vector from which SPD matrix R is formed 
            self._L_dims = np.sum(np.arange(1,self._hidden_dims[-1]+1))
            self._L = Dense(name='L',
                            weight_initializer=self._weight_initializer,
                            weight_regularizer=None,
                            bias_initializer=self._bias_initializer,
                            activation=None).build(input, [self._L_dims])

            R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

        # for backwards compatibility
        try:

            if self._R_activation is not None:

                R = self._R_activation(R)

        except:

            pass

        return z, R

    def _kalmanFiterLayer(self, input=None):

        z, R = input

        self._kf_results = self._kalman_filter.fit([z,R])

        z_hat_pri = tf.squeeze(self._kf_results['z_hat_pri'],axis=-1)
        z_hat_post = tf.squeeze(self._kf_results['z_hat_post'],axis=-1)

        return z_hat_pri, z_hat_post
    
    def _postKalmanFilterAffineLayer(self, input=None):

        assert input is not None

        return Dense(name='post_kf_affine',
                     weight_initializer=None,
                     weight_regularizer=None,
                     bias_initializer=self._bias_initializer,
                     activation=None).build(input, [self._hidden_dims[-2]])
        
    def _decoderLayer(self, input=None):

        assert input is not None
        
        return Dense(name='decoder',
                     weight_initializer=self._weight_initializer,
                     weight_regularizer=None,
                     bias_initializer=self._bias_initializer,
                     activation=self._activation).build(input, self._hidden_dims[::-1][2:]+[self._output_dim])
            
    def _outputLayer(self, input=None):

        assert input is not None
        
        return Dense(name='y_hat',
                     weight_initializer=self._weight_initializer,
                     weight_regularizer=None,
                     bias_initializer=self._bias_initializer,
                     activation=None).build(input, [self._output_dim])
     
    def _generate_spd_cov_matrix(self, R):

        """ 
        Generates a symmetric covariance matrix for the input covariance
        given input vector with first n_dims elements the diagonal and the
        remaining elements the off-diagonal elements
        """

        ################################
        # SPD Matrix Based on BPKF Paper
        ################################

        eps = 1e-1

        # initial upper triangular matrix
        L = tf.contrib.distributions.fill_triangular(R, upper = False)
        X = tf.matmul(L,L,transpose_b=True) + eps * tf.eye(tf.shape(L)[0],dtype=tf.float64)

        return X

    def _make_spd_matrix(self, x):

        """ Wrapper for sklearn make_spd_matrix """

        return tf.py_func(make_spd_matrix, [x], tf.float64)
    
    def _shapiro_wilk(self, x):

        x = tf.reshape(x, [tf.size(x)])
        print('X.shape in Shapiro-Wilk:%s' % (tf.shape(x)))

        # number of samples
        # n_samples = tf.shape(x, out_type=tf.float32)[0]
        n_samples = tf.to_float(tf.shape(x)[0])

        # sample range
        sample_range = tf.range(n_samples, name='sample_range')

        # m_i values
        m_i = tf.map_fn(lambda i: tf_ndtri(tf.divide(tf.subtract(tf.add(i,1.0),0.375),
        tf.add(n_samples,0.25))), sample_range)

        # m
        m = tf.cast(tf.reduce_sum(tf.square(m_i)), dtype=tf.float32)

        # u
        u = 1.0 / tf.sqrt(tf.cast(n_samples,dtype=tf.float32))

        # a[n-1]
        a_n_1 = tf.multiply(-3.582633,tf.pow(u,5)) + \
        tf.multiply(5.682633,tf.pow(u,4)) - \
        tf.multiply(1.752461,tf.pow(u,3)) - \
        tf.multiply(0.293762,tf.pow(u,2)) + \
        tf.multiply(0.042981,u) + \
        tf.multiply(m_i[-2],tf.pow(m,-0.5))

        # a[n]
        a_n = tf.multiply(-2.706056,tf.pow(u,5)) + \
        tf.multiply(4.434685,tf.pow(u,4)) - \
        tf.multiply(2.071190,tf.pow(u,3)) - \
        tf.multiply(0.147981,tf.pow(u,2)) + \
        tf.multiply(0.221156,u) + \
        tf.multiply(m_i[-1],tf.pow(m,-0.5))

        # stack first two values of a
        a = tf.stack([-a_n,-a_n_1])

        # epsilon
        epsilon = (m - 2.0*m_i[-1]**2 - 2.0*m_i[-2]**2) / (1.0 - 2.0*a_n**2 - 2.0*a_n_1**2)

        # compute a values
        a_range = tf.range(start=2.0, limit=n_samples-2.0, name='a_range')
        a_mid = tf.map_fn(lambda i: m_i[tf.to_int32(i)] / tf.sqrt(epsilon), a_range)

        # concat a values 2 through n-2
        a = tf.concat([a,a_mid],0)

        # concat last two a values
        a = tf.concat([a,tf.expand_dims(a_n_1,axis=0)],0)
        a = tf.concat([a,tf.expand_dims(a_n,axis=0)],0)

        # compute W
        # x = tf.Variable(np.sort(x), name='x', dtype=tf.float32)
        x, _ = tf.nn.top_k(x, k=tf.size(x), sorted=True)
        x = tf.cast(x, dtype=tf.float32)
        print('X.shape: %s' % (x.shape,))
        W_top = tf.square(tf.reduce_sum(tf.multiply(a,x)))
        W_bottom = tf.reduce_sum(tf.square(x - tf.reduce_mean(x)))

        W = tf.divide(W_top, W_bottom)

        # return W, a, epsilon, u, m, m_i
        return W

class HilbertAutoencoderKalmanFilter(AutoencoderKalmanFilter):

    """
    Orthogonal Polynomial-Autoencoder KalmanFilter Class
    """
    
    def __init__(self, params=None, kf_params=None):

        super().__init__(params=params, kf_params=kf_params)

    ##################
    # Public Methods #
    ##################
    
    ###################
    # Private Methods #
    ###################

    def _setPlaceholders(self):

        super()._setPlaceholders()
        self._support = tf.placeholder(dtype=tf.float64, shape=(1,self._input_dim), name='support')
        self._P = tf.squeeze(tf.map_fn(self._polyVector,self._support), axis=0)
        
    def _preKalmanFilterAffineLayer(self, input=None):

        assert input is not None
        
        z = Dense(name='z',
                  weight_initializer=self._weight_initializer,
                  weight_regularizer=None,
                  bias_initializer=self._bias_initializer,
                  activation=None).build(input, [self._hidden_dims[-1]])

        z = tf.expand_dims(z, axis=-1) 

        R  = Dense(name='R',
                  weight_initializer=self._weight_initializer,
                  weight_regularizer=None,
                  bias_initializer=self._bias_initializer,
                  activation=None).build(input, [self._hidden_dims[-1]])

        R = tf.expand_dims(tf.expand_dims(R,axis=-1),axis=-1)

        # learned noise covariance
        # if self._R_model == 'learned':

        #     # learn L, which is vector from which SPD matrix R is formed 
        #     self._L_dims = np.sum(np.arange(1,16+1))
        #     self._L = Dense(name='L',
        #                     weight_initializer=self._weight_initializer,
        #                     weight_regularizer=None,
        #                     bias_initializer=self._bias_initializer,
        #                     activation=None).build(input, [self._L_dims])

        #     R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

        # # identity noise covariance
        # elif self._R_model == 'identity':

        #     R = tf.eye(self._hidden_dims[-1], batch_shape=[tf.shape(self._z)[0]], dtype=tf.float64)

        if self._R_activation is not None:

            R = self._R_activation(R)

        return z, R
    
    def _kalmanFiterLayer(self, input=None):

        assert input is not None

        z, R = input

        # filter each trial in mini batch
        z_hat_pri, z_hat_post = tf.map_fn(self._filterEncoderOutput, [z,R], dtype=(tf.float64,tf.float64))

        return tf.squeeze(z_hat_pri,axis=-1), tf.squeeze(z_hat_post,axis=-1)
        
    def _filterEncoderOutput(self, input=None):

        assert input is not None

        z, R = input

        # transpose for Kalman Filter compatibility
        #input = tf.transpose(z)

        # add single dimension for Kalman Filter compatibility
        #input = tf.expand_dims(input, axis=1)

        output = super()._kalmanFiterLayer(input)

        return output
        
    def _outputLayer(self, input=None):

        assert input is not None
        
        c = Dense(name='c',
                  weight_initializer=self._weight_initializer,
                  weight_regularizer=None,
                  bias_initializer=self._bias_initializer,
                  activation=None).build(self._decoder, [self._output_dim])

        y_hat = tf.map_fn(lambda c : tf.squeeze(tf.matmul(c,tf.transpose(self._P)),axis=0), [c], dtype=tf.float64, name='y_hat')

        return y_hat
        
    def _polyVector(self,x):

        #N = self._y_hat.get_shape()[1]
        N = self._output_dim-1

        try:
        
            poly = Legendre(x,N).tensor
        
        except:
        
            x = tf.expand_dims(x,axis=1)
            poly = Legendre(x,N).tensor

        return poly
        
