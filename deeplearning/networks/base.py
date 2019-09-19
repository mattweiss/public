import os
from time import time
import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from abc import ABC, abstractmethod
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset

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
    def fit(self, dataset=None, save_model=False):

        pass
            
    @abstractmethod
    def predict(self, x=None):

        pass
        
    @abstractmethod
    def evaluate(self, x=None, y=None, t=None, save_results=True):

        pass
        
    def getModelSummary(self):

        pass
        
        # with tf.Session() as sess:
        
        #     variables_names = [v.name for v in tf.trainable_variables()]
        #     values = sess.run(variables_names)

        # for k, v in zip(variables_names, values):

        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)

        # # Print trainable variables
        # variables_names = list()

        # with tf.Session() as sess:
        
        #     for v in tf.trainable_variables():

        #         # add variables to list for printing
        #         variables_names.append( v.name )

        #     # print varibles and shape
        #     values = sess.run(variables_names)

        # for k, v in zip(variables_names, values):
                
        #     print ("Trainable Variable: %s %s" % (k, v,))

    def getAttributeNames(self):

        return self.__dict__.keys()
    
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

    def fit(self, dataset=None, save_model=False):
        
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

    def fitDomainRandomization(self, dr_params=None, save_model=False):

        # create domainRandomizationDataset object
        self._dr_dataset = DomainRandomizationDataset(dr_params)

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())

            # start time
            start_time = time()
            
            for epoch in range(1, self._epochs+1):

                # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
                dr_data = self._dr_dataset.generateDataset()

                # train on all trials
                for x_train, y_train in zip(dr_data['x_train'],dr_data['y_train']):

                    # training op
                    _ = sess.run(self._optimizer_op, feed_dict={self._X:x_train, self._y:y_train})

                # train and val loss lists
                train_loss = list()
                val_loss = list()

                # compute loss on all training and validation trials
                for x_train, y_train, x_val, y_val in zip(dr_data['x_train'],dr_data['y_train'],dr_data['x_val'],dr_data['y_val']):

                    # loss op
                    train_loss.append(sess.run(self._loss_op, feed_dict={self._X:x_train, self._y:y_train}))
                    val_loss.append(sess.run(self._loss_op, feed_dict={self._X:x_val, self._y:y_val}))

                self._history['train_loss'].append(np.asarray(train_loss).mean())
                self._history['val_loss'].append(np.asarray(val_loss).mean())

                if len(self._history['train_loss']) > self._history_size:

                    self._history['train_loss'].pop(0)
                    self._history['val_loss'].pop(0)
                
                print('Epoch {epoch} training loss {train_loss} Val Loss {val_loss}'.format(epoch=epoch, train_loss=self._history['train_loss'][-1], val_loss=self._history['val_loss'][-1]))

                # if epoch == self._epochs:

                #     # plot using last x_train and x_val
                #     train_pred, z_hat_post_train, z_train = sess.run([self._y_hat,self._z_hat_post,self._z], feed_dict={self._X:x_train})
                #     val_pred, z_hat_post_val, z_val = sess.run([self._y_hat,self._z_hat_post,self._z], feed_dict={self._X:x_val})

                #     plt.figure(figsize=(12,12))

                #     plt.subplot(221)
                #     plt.scatter(range(x_train.shape[0]), x_train, label='train', color='green')
                #     plt.plot(y_train, label='train_gt')
                #     plt.plot(train_pred, label='train_pred')
                #     plt.grid()
                #     plt.legend()

                #     plt.subplot(222)
                #     plt.scatter(range(x_val.shape[0]), x_val, label='val', color='green')
                #     plt.plot(y_val, label='val_gt')
                #     plt.plot(val_pred, label='val_pred')
                #     plt.grid()
                #     plt.legend()
                    
                #     plt.subplot(223)
                #     for dim in range(z_hat_post_train.shape[1]):
                        
                #         plt.plot(range(z_train.shape[0]), z_hat_post_train[:,dim], label='z_hat_post_train', color='green')
                #         plt.scatter(range(z_train.shape[0]), z_train[:,dim], label='z_train')

                #     plt.grid()
                #     #plt.legend()
                    
                #     plt.subplot(224)
                #     for dim in range(z_hat_post_val.shape[1]):
                        
                #         plt.plot(range(z_val.shape[0]), z_hat_post_val[:,dim], label='z_hat_post_val', color='green')
                #         plt.scatter(range(z_val.shape[0]), z_val[:,dim], label='z_val')
                        
                #     plt.grid()
                #     #plt.legend()
                    
                #     plt.show()
                #     plt.close()

            self._history['runtime'] = (time() - start_time) / 60.0

            if save_model:

                self._saveModel(sess)

        return self._history

            
    def predict(self, dataset=None):

        pass
        
    def evaluate(self, x=None, y=None, t=None, save_results=True):

        assert x is not None
        assert y is not None
        assert t is not None

        # model predictions
        x_hat_list = list()
        
        # load trained model
        model_results_path = './results/tensorflow_model.ckpt'
        
        with tf.Session() as sess:
            
            tf.train.Saver().restore(sess, model_results_path)

            for X,Y in zip(x,y):

                test_loss, x_hat = sess.run([self._loss_op,self._y_hat], feed_dict={self._X:X, self._y:Y})
                self._history['test_loss'].append(test_loss)
                x_hat_list.append(x_hat)
                
        x_hat = np.asarray(x_hat_list)

        # save predictions
        if save_results:

            test_results_dict = {
                'x':x,
                'y':y,
                'x_hat':x_hat,
                't':t
                }
            
            saveDict(save_dict=test_results_dict, save_path='./results/testing_results.pkl')
            
        return self._history
                
    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

        self._y_hat = self._buildDenseLayers(self._X, self._hidden_dims)

    def _setLoss(self):
            
        self._loss_op = tf.cast(self._loss(self._y, self._y_hat), tf.float64) + tf.cast(tf.losses.get_regularization_loss(), tf.float64)
            
    def _setOptimizer(self):

        if self._optimizer.__name__ == 'AdamOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate).minimize(self._loss_op)

        elif self._optimizer.__name__ == 'MomentumOptimizer':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate, momentum=self._momentum).minimize(self._loss_op)

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
            
        return output

    def _generateMinibatches(self, X, y):

        X_mb = [X[i * self._mbsize:(i + 1) * self._mbsize,:] for i in range((X.shape[0] + self._mbsize - 1) // self._mbsize )]
        y_mb = [y[i * self._mbsize:(i + 1) * self._mbsize] for i in range((y.shape[0] + self._mbsize - 1) // self._mbsize )]

        return X_mb, y_mb

    def _saveModel(self, tf_session=None):

        assert tf_session is not None
        
        # save Tensorflow variables

        # name of file weights are saved to
        self._trained_model_file = os.getcwd() + self._results_dir + 'tensorflow_model.ckpt'

        # save everything
        tf.train.Saver().save(tf_session, self._trained_model_file)
