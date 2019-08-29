import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pdb import set_trace as st

from abc import ABC, abstractmethod
from dovebirdia.utilities.base import dictToAttributes

class AbstractNetwork(ABC):

    """
    Abstract base class for network
    """

    def __init__(self, params=None):

        """ 
        TODO: Add parameter list
        """

        dictToAttributes(self,params)
        
        # hold, etc.
        self._history = {
            'train_loss':list(),
            'val_loss':list(),
            'test_loss':list(),
            }
        
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


    ##################
    # Public Methods #
    ##################
            
    @abstractmethod
    def fit(self, dataset=None, save_weights=False):

        pass
            
    @abstractmethod
    def predict(self, dataset=None):

        pass
        
    @abstractmethod
    def evaluate(self, dataset=None):

        pass
    
    def getModelSummary(self):

        # Print trainable variables
        variables_names = list()

        with tf.Session() as sess:
        
            for v in tf.trainable_variables():

                # add variables to list for printing
                variables_names.append( v.name )

            # print varibles and shape
            values = sess.run(variables_names)

        for k, v in zip(variables_names, values):
                
            print ("Trainable Variable: %s %s" % (k, v.shape,))

    def getAttributeNames(self):

        return self.__dict__.keys()
    
    ###################
    # Private Methods #
    ###################

    @abstractmethod
    def _buildNetwork(self):

        pass
    
    @abstractmethod
    def _setOptimizer(self):

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

    def fit(self, dataset=None, save_weights=False):
        
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
                
        if save_weights:

            pass

    def fitDomainRandomization(self, fns=None, save_weights=False):

        dictToAttributes(self,fns)

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            for epoch in range(1, self._epochs+1):
            
                # randomly select one of the functions in self._fns
                self._fn_name, self._fn_def, self._fn_params = random.choice(self._fns)

                # generate training and validation curves
                x_train, y_train = self._generateDomainRandomizationData(self._fn_params)
                x_val, y_val = self._generateDomainRandomizationData(self._fn_params)

                # training op
                _ = sess.run(self._optimizer_op, feed_dict={self._X:x_train, self._y:y_train})
                # loss op
                train_loss = sess.run(self._loss_op, feed_dict={self._X:x_train, self._y:y_train})
                val_loss = sess.run(self._loss_op, feed_dict={self._X:x_val, self._y:y_val})

                self._history['train_loss'].append(train_loss)
                self._history['val_loss'].append(val_loss)

                if len(self._history['train_loss']) > self._history_size:

                    self._history['train_loss'].pop(0)
                    self._history['val_loss'].pop(0)
                
                print('Epoch {epoch} training loss {train_loss} Val Loss {val_loss}'.format(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

                # if epoch % 100 == 0:

                #     train_pred = sess.run(self._X_hat, feed_dict={self._X:x_train})
                #     val_pred = sess.run(self._X_hat, feed_dict={self._X:x_val})

                #     plt.figure(figsize=(12,6))
                #     plt.subplot(121)
                #     plt.scatter(range(x_train.shape[0]), x_train, label='train', color='green')
                #     plt.plot(y_train, label='train_gt')
                #     plt.plot(train_pred, label='train_pred')
                #     plt.grid()
                #     plt.legend()
                #     plt.subplot(122)
                #     plt.scatter(range(x_val.shape[0]), x_val, label='val', color='green')
                #     plt.plot(y_val, label='val_gt')
                #     plt.plot(val_pred, label='val_pred')
                #     plt.grid()
                #     plt.legend()
                #     plt.show()
                #     plt.close()

        if save_weights:

            pass

        return self._history

            
    def predict(self, dataset=None):

        pass
        
    def evaluate(self, dataset=None):

        pass
    
    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

        self._X_hat = self._buildDenseLayers(self._X, self._hidden_dims)

    def _setLoss(self):
            
        self._loss_op = tf.cast(self._loss(self._y, self._X_hat), tf.float64) + tf.cast(tf.losses.get_regularization_loss(), tf.float64)
            
    def _setOptimizer(self):

        if self._optimizer.__name__ == 'AdamOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)

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
                                           kernel_initializer=self._kernel_initializer,
                                           bias_initializer=self._bias_initializer,
                                           kernel_regularizer=self._kernel_regularizer,
                                           bias_regularizer=self._bias_regularizer,
                                           activity_regularizer=self._activity_regularizer,
                                           kernel_constraint=self._kernel_constraint,
                                           bias_constraint=self._bias_constraint)(output)

        output = tf.keras.layers.Dense(units=self._output_dim,
                                       activation=None,
                                       use_bias=self._use_bias,
                                       kernel_initializer=self._kernel_initializer,
                                       bias_initializer=self._bias_initializer,
                                       kernel_regularizer=self._kernel_regularizer,
                                       bias_regularizer=self._bias_regularizer,
                                       activity_regularizer=self._activity_regularizer,
                                       kernel_constraint=self._kernel_constraint,
                                       bias_constraint=self._bias_constraint)(output)
            
        return output

    def _generateMinibatches(self, X, y):

        X_mb = [X[i * self._mbsize:(i + 1) * self._mbsize,:] for i in range((X.shape[0] + self._mbsize - 1) // self._mbsize )]
        y_mb = [y[i * self._mbsize:(i + 1) * self._mbsize] for i in range((y.shape[0] + self._mbsize - 1) // self._mbsize )]

        return X_mb, y_mb
        
    def _generateDomainRandomizationData(self, params):

        param_list = list()

        for param in params:

            if isinstance(param, tuple):

                param_list.append(np.random.uniform(param[0], param[1]))
                
            else:

                param_list.append(param)
                
        x = np.linspace(self._x_range[0], self._x_range[1], self._n_samples)
        y = self._fn_def(x, *param_list)
        y_noise = y + self._noise(**self._noise_params, size=self._n_samples)

        return np.expand_dims(y_noise, axis=-1), np.expand_dims(y, axis=-1)
