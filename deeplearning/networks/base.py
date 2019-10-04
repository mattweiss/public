import os
from time import time
import copy
import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from abc import ABC, abstractmethod
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.deeplearning.layers.base import DenseLayer
from dovebirdia.deeplearning.regularizers.base import orthonormal_regularizer

try:

    import matplotlib.pyplot as plt

except:

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    
class AbstractNetwork(ABC):

    """
    Abstract base class for network
    """

    def __init__(self, params=None):

        assert isinstance(params,dict)

        dictToAttributes(self,params)

        # backwards compatibility
        try:

            self._hidden_layer_dict = {

            'weight_initializer':self.__dict__['_weight_initializer'],
            'weight_regularizer':self.__dict__['_weight_regularizer'],
            'bias_initializer':self.__dict__['_bias_initializer'],
            'bias_regularizer':self.__dict__['_bias_regularizer'],
            'activation':self.__dict__['_activation'],
            'use_bias':self.__dict__['_use_bias'],

        }

        except:

           self._hidden_layer_dict = {

            'weight_initializer':self.__dict__['_weight_initializer'],
            'weight_regularizer':self.__dict__['_weight_regularizer'],
            'bias_initializer':self.__dict__['_bias_initializer'],
            'bias_regularizer':self.__dict__['_bias_regularizer'],
            'activation':self.__dict__['_activation'],
            'use_bias':self.__dict__['_use_bias'],

        }
            
        self._affine_layer_dict = copy.deepcopy(self._hidden_layer_dict)
        self._affine_layer_dict['activation'] = None

        self._z_layer_dict = copy.deepcopy(self._hidden_layer_dict)
        self._z_layer_dict['activation'] = None
        
        self._output_layer_dict = copy.deepcopy(self._hidden_layer_dict)
        self._output_layer_dict['activation'] = self._output_activation
        
        # hold, etc.
        self._history = {
            'train_loss':list(),
            'val_loss':list(),
            'test_loss':list(),
            }
        
    ##################
    # Public Methods #
    ##################

    def compile(self):

        """ 
        Build Network
        """

        ############################
        # Build Network
        ############################
        self._buildNetwork()
        
        ############################
        # Set Optimizer
        ############################
        self._setLoss()
        
        ############################
        # Set Optimizer
        ############################
        self._setOptimizer()

    @abstractmethod
    def fit(self, dataset=None, save_model=False):

        pass
            
    @abstractmethod
    def predict(self, x=None):

        pass
        
    @abstractmethod
    def evaluate(self, x=None, y=None, t=None, save_results=True):

        pass
    
    ###################
    # Private Methods #
    ###################
        
    @abstractmethod
    def _buildNetwork(self):

        pass
        
    @abstractmethod
    def _setLoss(self):

        pass
    
    @abstractmethod
    def _setOptimizer(self):

        pass

    @abstractmethod
    def _saveModel(self):

        pass
    
class FeedForwardNetwork(AbstractNetwork):

    """
    Feed Forward Network Class
    """

    def __init__(self, params=None):

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    def fit(self, dataset=None, dr_params=None, save_model=False):

        if dataset is not None:

            return self._fit(dataset, save_model)

        elif dr_params is not None:

            return self._fitDomainRandomization(dr_params, save_model)
            
    def predict(self, dataset=None):

        pass
        
    def evaluate(self, x=None, y=None, t=None, save_results=True):

        pass
                
    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None), name='y')

        self._y_hat = DenseLayer(self.__dict__).build(self._X, self._hidden_dims, scope='layers')
    
    def _setLoss(self):

        self._loss_op = tf.cast(self._loss(self._y, self._y_hat), tf.float64) + tf.cast(tf.losses.get_regularization_loss(), tf.float64)
        
    def _setOptimizer(self):

        if self._optimizer.__name__ == 'AdamOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)

        elif self._optimizer.__name__ == 'MomentumOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate, momentum=self._momentum).minimize(self._loss_op)

    def _fit(self, dataset=None, save_model=False):
        
        dictToAttributes(self,dataset)

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())
                
            # loop over epochs
            for epoch in range(1, self._epochs+1):

                # minibatch split
                self._x_train_mb, self._y_train_mb = self._generateMinibatches(self._x_train, self._y_train)

                # loop over minibatches
                for x_mb, y_mb in zip(self._x_train_mb, self._y_train_mb):
                
                    # training op
                    _ = sess.run(self._optimizer_op, feed_dict={self._X:x_mb, self._y:y_mb})

                    # loss op
                    train_loss = sess.run(self._loss_op, feed_dict={self._X:x_mb, self._y:y_mb})

               # validation loss
                val_loss = sess.run(self._loss_op, feed_dict={self._X:self._x_val, self._y:self._y_val})

                print('Epoch {epoch} Training Loss {train_loss} Val Loss {val_loss}'.format(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
                
        if save_model:

            self._saveModel()

    def _fitDomainRandomization(self, dr_params=None, save_model=False):

        # create domainRandomizationDataset object
        self._dr_dataset = DomainRandomizationDataset(dr_params)

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            variables_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)

            for k, v in zip(variables_names, values):

                print(v.shape, k)

            # start time
            start_time = time()
            
            for epoch in range(1, self._epochs+1):

                # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
                dr_data = self._dr_dataset.generateDataset()
                
                # train and val loss lists
                train_loss = list()
                val_loss = list()
                
                # train on all trials
                for x_train, y_train, x_val, y_val in zip(dr_data['x_train'],dr_data['y_train'],dr_data['x_val'],dr_data['y_val']):

                    # training op
                    _ = sess.run(self._optimizer_op, feed_dict={self._X:x_train, self._y:y_train})

                    # loss op
                    train_loss.append(sess.run(self._loss_op, feed_dict={self._X:x_train, self._y:y_train}))
                    val_loss.append(sess.run(self._loss_op, feed_dict={self._X:x_val, self._y:y_val}))
                    
                self._history['train_loss'].append(np.asarray(train_loss).mean())
                self._history['val_loss'].append(np.asarray(val_loss).mean())

                if len(self._history['train_loss']) > self._history_size:

                    self._history['train_loss'].pop(0)
                    self._history['val_loss'].pop(0)
                
                print('Epoch {epoch} training loss {train_loss} Val Loss {val_loss}'.format(epoch=epoch,
                                                                                            train_loss=self._history['train_loss'][-1],
                                                                                            val_loss=self._history['val_loss'][-1]))

                #if epoch == self._epochs:

                # train_pred, z_hat_post_train, z_train = sess.run([self._y_hat,self._z_hat_post,self._z], feed_dict={self._X:x_train})
                # val_pred, z_hat_post_val, z_val = sess.run([self._y_hat,self._z_hat_post,self._z], feed_dict={self._X:x_val})

                # plt.figure(figsize=(12,6))

                # plt.subplot(121)

                # for sensor in range(x_train.shape[1]):

                #     plt.scatter(range(x_train.shape[0]), x_train[:,sensor], label='train', color='green')
                #     plt.plot(y_train[:,sensor], label='train_gt')
                #     plt.plot(train_pred[:,sensor], label='train_pred')

                # plt.grid()
                # plt.legend()

                # plt.subplot(122)

                # for sensor in range(x_val.shape[1]):

                #     plt.scatter(range(x_val.shape[0]), x_val[:,sensor], label='val', color='green')
                #     plt.plot(y_val[:,sensor], label='val_gt')
                #     plt.plot(val_pred[:,sensor], label='val_pred')

                # plt.grid()
                # plt.legend()

                # plt.subplot(223)
                # for dim in range(z_hat_post_train.shape[1]):

                #     plt.plot(range(z_train.shape[0]), z_hat_post_train[:,dim], label='z_hat_post_train', color='green')
                #     plt.scatter(range(z_train.shape[0]), z_train[:,dim], label='z_train')

                # plt.grid()
                # plt.legend()

                # plt.subplot(224)
                # for dim in range(z_hat_post_val.shape[1]):

                #     plt.plot(range(z_val.shape[0]), z_hat_post_val[:,dim], label='z_hat_post_val', color='green')
                #     plt.scatter(range(z_val.shape[0]), z_val[:,dim], label='z_val')

                # plt.grid()
                # plt.legend()

                #plt.savefig('./new_code_{epoch}'.format(epoch=epoch))
                # plt.show()
                #plt.close()

            self._history['runtime'] = (time() - start_time) / 60.0

            if save_model:

                self._saveModel(sess)

        return self._history
    
    def _generateMinibatches(self, X, y):

        X_mb = [X[i * self._mbsize:(i + 1) * self._mbsize,:] for i in range((X.shape[0] + self._mbsize - 1) // self._mbsize )]
        y_mb = [y[i * self._mbsize:(i + 1) * self._mbsize] for i in range((y.shape[0] + self._mbsize - 1) // self._mbsize )]

        return X_mb, y_mb

    def _saveModel(self, tf_session=None):

        assert tf_session is not None
        
        # save Tensorflow variables

        # name of file weights are saved to
        self._trained_model_file = os.getcwd() + self._results_dir + 'trained_model.ckpt'

        # save everything
        tf.train.Saver().save(tf_session, self._trained_model_file)

    def _buildDenseLayers(self, input=None, hidden_dims=None, name=None):

        assert input is not None
        assert hidden_dims is not None

        # loop over hidden layers
        for dim_index, dim in enumerate(hidden_dims):

            # pass input parameter on first pass
            output = input if dim_index == 0 else output

            # hidden layer
            output = tf.keras.layers.Dense(units=dim,
                                           activation=self._activation,
                                           use_bias=self._use_bias,
                                           kernel_initializer=self._weight_initializer,
                                           bias_initializer=self._bias_initializer,
                                           kernel_regularizer=self._weight_regularizer,
                                           bias_regularizer=self._bias_regularizer,
                                           activity_regularizer=self._activity_regularizer,
                                           kernel_constraint=self._weight_constraint,
                                           bias_constraint=self._bias_constraint)(output)

            print(output)
            
        return output
