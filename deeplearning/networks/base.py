import os
from time import time
#import copy
import numpy as np
from scipy import stats
import tensorflow as tf
from pdb import set_trace as st

from abc import ABC, abstractmethod
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.deeplearning.layers.base import Dense
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

        self._y_hat = Dense(self.__dict__).build(self._X, self._hidden_dims, scope='layers')
    
    def _setLoss(self):

        self._loss_op = tf.cast(self._loss(self._y, self._y_hat), tf.float64) + tf.cast(tf.losses.get_regularization_loss(), tf.float64)

    def _setOptimizer(self):

        if self._optimizer.__name__ == 'AdamOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)

        elif self._optimizer.__name__ == 'MomentumOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate, momentum=self._momentum).minimize(self._loss_op)

        elif self._optimizer.__name__ == 'GradientDescentOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)
            
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

            self._history['p_value'] = list()
            
            for epoch in range(1, self._epochs+1):

                # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
                dr_data = self._dr_dataset.generateDataset()
                
                # train and val loss lists
                train_loss = list()
                val_loss = list()

                support = np.linspace(-1,1,100).reshape(1,-1)
                
                # train on all trials
                for x_train, y_train, x_val, y_val in zip(dr_data['x_train'],dr_data['y_train'],dr_data['x_val'],dr_data['y_val']):

                    train_feed_dict = {self._X:x_train, self._y:y_train}
                    val_feed_dict = {self._X:x_val, self._y:y_val}

                    # OAEKF
                    if hasattr(self, '_support'):

                        train_feed_dict[self._support] = support
                        val_feed_dict[self._support] = support
                        
                    # training op
                    _ = sess.run(self._optimizer_op, feed_dict=train_feed_dict)
                    
                    # loss op
                    train_loss.append(sess.run(self._loss_op, feed_dict=train_feed_dict))
                    val_loss.append(sess.run(self._loss_op, feed_dict=val_feed_dict))
                    
                self._history['train_loss'].append(np.asarray(train_loss).mean())
                self._history['val_loss'].append(np.asarray(val_loss).mean())

                if len(self._history['train_loss']) > self._history_size:

                    self._history['train_loss'].pop(0)
                    self._history['val_loss'].pop(0)

                print('Epoch {epoch} training loss {train_loss:.4f} Val Loss {val_loss:.4f}'.format(epoch=epoch,
                                                                                                    train_loss=self._history['train_loss'][-1],
                                                                                                    val_loss=self._history['val_loss'][-1]))
            
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
