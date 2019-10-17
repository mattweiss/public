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
        self._encoder = Dense(self._hidden_layer_dict).build(self._X, self._hidden_dims[:-1], scope='encoder')
        self._decoder = Dense(self._hidden_layer_dict).build(self._encoder, self._hidden_dims[::-1][1:], scope='decoder')
        
        # output layer
        self._y_hat = Dense(self._affine_layer_dict).build(self._decoder, [self._output_dim], scope='output')
        
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

                test_loss, x_hat, z, z_hat_post, R = sess.run([self._loss_op,self._y_hat,self._z,self._z_hat_post,self._R], feed_dict={self._X:X, self._y:Y})
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

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')

        # weight regularizer
        try:

            weight_regularizer = self._weight_regularizer(self._weight_regularizer_scale)

        except:

            weight_regularizer = self._weight_regularizer
        
        # encoder      
        self._encoder = Dense(name='encoder',
                              weight_initializer=self._weight_initializer,
                              weight_regularizer=weight_regularizer,
                              bias_initializer=self._bias_initializer,
                              activation=self._activation).build(self._X, self._hidden_dims[:-1])

        self._z = Dense(name='z',
                        weight_initializer=self._weight_initializer,
                        weight_regularizer=weight_regularizer,
                        bias_initializer=self._bias_initializer,
                        activation=None).build(self._encoder, [self._hidden_dims[-1]])
        
        try:
        
            # learned noise covariance
            if self._R_model == 'learned':

                # learn L, which is vector from which SPD matrix R is formed 
                self._L_dims = np.sum( np.arange( 1, self._hidden_dims[-1] + 1 ) )
                self._L = Dense(name='L',
                                weight_initializer=self._weight_initializer,
                                weight_regularizer=weight_regularizer,
                                bias_initializer=self._bias_initializer,
                                activation=None).build(self._encoder, [self._L_dims])

                self._R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

            elif self._R_model == 'identity':

                self._R = tf.eye(self._hidden_dims[-1], batch_shape=[tf.shape(self._z)[0]], dtype=tf.float64)
                
        except:

            # learn L, which is vector from which SPD matrix R is formed 
            self._L_dims = np.sum( np.arange( 1, self._hidden_dims[-1] + 1 ) )
            self._L = Dense(name='L',
                            weight_initializer=self._weight_initializer,
                            weight_regularizer=weight_regularizer,
                            bias_initializer=self._bias_initializer,
                            activation=self._activation).build(self._encoder, [self._L_dims])

            self._R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

            # for backwards compatibility
            try:

                if self._R_activation is not None:

                    self._R = self._R_activation(self._R)

            except:

                pass
            
        # Kalman Filter a priori measurement estimate
        self._kf_results = self._kalman_filter.fit([self._z,self._R])
        self._z_hat_pri = tf.squeeze(self._kf_results['z_hat_pri'],axis=-1)
        self._z_hat_post = tf.squeeze(self._kf_results['z_hat_post'],axis=-1)
        
        self._post_kf_affine = Dense(name='post_kf_affine',
                                     weight_initializer=self._weight_initializer,
                                     weight_regularizer=weight_regularizer,
                                     bias_initializer=self._bias_initializer,
                                     activation=self._activation).build(self._z_hat_pri, [self._hidden_dims[-2]])

        self._decoder = Dense(name='decoder',
                              weight_initializer=self._weight_initializer,
                              weight_regularizer=weight_regularizer,
                              bias_initializer=self._bias_initializer,
                              activation=self._activation).build(self._post_kf_affine, self._hidden_dims[::-1][2:]+[self._output_dim])

        self._y_hat = Dense(name='y_hat',
                            weight_initializer=self._weight_initializer,
                            weight_regularizer=weight_regularizer,
                            bias_initializer=self._bias_initializer,
                            activation=None).build(self._decoder, [self._output_dim])
        
        # N=3
        # c = tf.get_variable(name='c',
        #                     shape=(N+1,self._output_dim),
        #                     initializer=self._weight_initializer,
        #                     trainable=True,
        #                     dtype=tf.float64)
        
        # P = Legendre(self._decoder, N).tensor
        # self._y_hat = tf.matmul(P,c)

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

class OrthoKalmanFilter(AutoencoderKalmanFilter):

    """
    Orthogonal Polynomial-KalmanFilter Class
    """
    
    def __init__(self, params=None, kf_params=None):

        super().__init__(params=params, kf_params=kf_params)

        ###################
    # Private Methods #
    ###################
  
    def _buildNetwork(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')

        # weight regularizer
        try:

            weight_regularizer = self._weight_regularizer(self._weight_regularizer_scale)

        except:

            weight_regularizer = self._weight_regularizer
            
        N=4
        c_pre = tf.get_variable(name='c_pre',
                            shape=(N+1,self._input_dim),
                            initializer=self._weight_initializer,
                            trainable=True,
                            dtype=tf.float64)
        
        P_pre = Legendre(self._X, N).tensor
        self._z = tf.matmul(P_pre,c_pre)

        try:
        
            # learned noise covariance
            if self._R_model == 'learned':

                # learn L, which is vector from which SPD matrix R is formed 
                self._L_dims = np.sum( np.arange( 1, self._input_dim + 1 ) )
                self._L = Dense(name='L',
                                weight_initializer=self._weight_initializer,
                                weight_regularizer=weight_regularizer,
                                bias_initializer=self._bias_initializer,
                                activation=self._activation).build(self._z, [self._L_dims])

                self._R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

            elif self._R_model == 'identity':

                self._R = tf.eye(self._hidden_dims[-1], batch_shape=[tf.shape(self._z)[0]], dtype=tf.float64)

            elif self._R_model == 'random':

                dims = tf.fill(dims=(tf.shape(self._z)[0],), value=self._hidden_dims[-1])
                self._R = tf.map_fn(self._make_spd_matrix, dims, dtype=tf.float64)

        except:

            # learn L, which is vector from which SPD matrix R is formed 
            self._L_dims = np.sum( np.arange( 1, self._input_dim + 1 ) )
            self._L = Dense(name='L',
                            weight_initializer=self._weight_initializer,
                            weight_regularizer=weight_regularizer,
                            bias_initializer=self._bias_initializer,
                            activation=self._activation).build(self._encoder, [self._L_dims])

            self._R = tf.map_fn(self._generate_spd_cov_matrix, self._L)
                
        # Kalman Filter a priori measurement estimate
        self._kf_results = self._kalman_filter.fit([self._z,self._R])
        self._z_hat_pri = tf.squeeze(self._kf_results['z_hat_pri'],axis=-1)
        self._z_hat_post = tf.squeeze(self._kf_results['z_hat_post'],axis=-1)

        c_post = tf.get_variable(name='c_post',
                            shape=(N+1,self._input_dim),
                            initializer=self._weight_initializer,
                            trainable=True,
                            dtype=tf.float64)

        P_post = Legendre(self._z_hat_pri, N).tensor
        self._y_hat = tf.matmul(P_post,c_post)
