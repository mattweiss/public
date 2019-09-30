import numpy as np
import tensorflow as tf
from scipy import stats
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.layers.base import DenseLayer

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
        self._encoder = DenseLayer(self._hidden_layer_dict).build(self._X, self._hidden_dims[:-1], scope='encoder')
        self._decoder = DenseLayer(self._hidden_layer_dict).build(self._encoder, self._hidden_dims[::-1][1:], scope='decoder')
        
        # output layer
        self._y_hat = DenseLayer(self._affine_layer_dict).build(self._decoder, [self._output_dim], scope='output')
        
class AutoencoderKalmanFilter(Autoencoder):

    """
    Autoencoder-KalmanFilter Class
    """
    
    def __init__(self, params=None, kf_params=None):

        # instantiate Kalman Filter before parent constructor as
        # the parent calls _buildNetwork()
        self._kalman_filter = KalmanFilter(params=kf_params)

        ######################################
        # KalmanFilter Used Before Dovebirdia
        ######################################
        # kf_params['dt'] = 1.0
        # dt = kf_params['dt']
        # q = kf_params['q']
        # h = kf_params['h']
        # latent_dims = kf_params['n_signals']
        # kf_params['F'] = np.kron(np.eye(latent_dims,dtype=np.float64), np.array([[1.0,dt],[0.0,1.0]],dtype=np.float64))
        # kf_params['Q'] = np.kron(np.eye(latent_dims,dtype=np.float64), np.array([[q,0.0],[0.0,q]],dtype=np.float64))
        # kf_params['H'] = np.kron(np.eye(latent_dims,dtype=np.float64), np.array( [ [1.0,0.0] ],dtype=np.float64))
        # kf_params['mb_size'] = 110
        # kf_params['n_samples'] = 110
        # kf_params['x0'] = np.zeros((kf_params['dimensions'][1]*latent_dims,1), dtype=np.float64)
        # #kf_params['P0'] = make_spd_matrix( kf_params['dimensions'][1]*latent_dims )
        # kf_params['P0'] = np.eye( kf_params['dimensions'][1] * latent_dims, dtype=np.float64 )
        # kf_params['K0'] = np.zeros( shape = kf_params['H'].T.shape, dtype = np.float64 )
        # self._my_kf_params = kf_params
        
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
        self._history['sw'] = list()
        
        with tf.Session() as sess:

            # backwards compatibility
            try:

                model_results_path = './results/trained_model.ckpt'
                tf.train.Saver().restore(sess, model_results_path)

            except:

                model_results_path = './results/tensorflow_model.ckpt'
                tf.train.Saver().restore(sess, model_results_path)

            for X,Y in zip(x,y):

                test_loss, x_hat, z, z_hat_post = sess.run([self._loss_op,self._y_hat,self._z,self._z_hat_post], feed_dict={self._X:X, self._y:Y})
                self._history['test_loss'].append(test_loss)
                x_hat_list.append(x_hat)
                z_list.append(z)
                z_hat_post_list.append(z_hat_post)

                for sensor in range(z.shape[1]):
                    
                    w,p = stats.shapiro(np.subtract(z[:,sensor],z_hat_post[:,sensor]))
                    self._history['sw'].append(p)
                
        x_hat = np.asarray(x_hat_list)
        z = np.asarray(z)
        z_hat_post = np.asarray(z_hat_post_list)
        
        # save predictions
        if save_results:

            test_results_dict = {
                'x':x,
                'y':y,
                'x_hat':x_hat,
                't':t,
                'z':z,
                'z_hat_post':z_hat_post,
                }
            
            saveDict(save_dict=test_results_dict, save_path='./results/testing_results.pkl')
            
        return self._history
    
    ###################
    # Private Methods #
    ###################
    
    def _buildNetwork(self):
        
        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')

        # encoder
        self._encoder = DenseLayer(self._hidden_layer_dict).build(self._X, self._hidden_dims, scope='encoder')

        # learn z
        self._z = DenseLayer(self._affine_layer_dict).build(self._encoder, [self._hidden_dims[-1]], scope='z')

        # learn L, which is vector from which SPD matrix R is formed 
        self._L_dims = np.sum(np.arange(1, self._hidden_dims[-1] + 1))
        self._L = DenseLayer(self._L_layer_dict).build(self._encoder, [self._L_dims], scope='L')
        
        # learned noise covariance
        self._R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

        ######################################
        # KalmanFilter Used Before Dovebirdia
        ######################################
        # self._my_kf_params['R'] = self._R
        # self._kalman_filter = KalmanFilter(self._my_kf_params)        
        # kf_results = self._kalman_filter.filter( self._z, fixed_R = False )
        # self._z_hat_pri = kf_results['z_hat_apri']           
        # self._z_hat_post = kf_results['z_hat_apost']
        # Kalman Filter a priori measurement estimate

        self._kf_results = self._kalman_filter.fit([self._z,self._R])
        self._z_hat_pri = tf.squeeze(self._kf_results['z_hat_pri'],axis=-1)
        self._z_hat_post = tf.squeeze(self._kf_results['z_hat_post'],axis=-1)

        # post kf affine transformation
        self._post_kf_affine = DenseLayer(self._affine_layer_dict).build(self._z_hat_pri, [self._hidden_dims[-1]], scope='post_kf_affine')

        # decoder
        self._decoder = DenseLayer(self._hidden_layer_dict).build(self._post_kf_affine, self._hidden_dims[::-1][1:], scope='decoder', dropout_rate=self._dropout_rate)

        # output layer
        self._y_hat = DenseLayer(self._output_layer_dict).build(self._decoder, [self._output_dim], scope='y_hat')
        
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
