import os, socket
from time import time
import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from keras import backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM as LSTM_LAYER
from keras.models import Sequential, load_model
from keras import optimizers, regularizers, initializers, losses

from dovebirdia.deeplearning.networks.base import AbstractNetwork, FeedForwardNetwork
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.datasets.outliers import generate_outliers

machine = socket.gethostname()
if machine == 'pengy':

    import matplotlib.pyplot as plt

else:

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
  
class LSTM(FeedForwardNetwork):

    """
    LSTM Class
    """

    def __init__(self, params=None):

        assert isinstance(params,dict)
        
        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    def evaluate(self,x=None,y=None,labels=None,
                 eval_ops=None,
                 attributes=None,
                 save_results=None):

        x, y, _ = self._generateDataset(x,y)

        return super().evaluate(x=x,y=y,attributes=attributes,save_results=save_results)
        
    ###################
    # Private Methods #
    ###################

    def _fitDomainRandomization(self, dr_params=None, save_model=False):

        # create domainRandomizationDataset object
        self._dr_dataset = DomainRandomizationDataset(dr_params)

        # dictionaries to hold training and validation data
        train_feed_dict = dict()
        val_feed_dict = dict()

        start_time = time()

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(1, self._epochs+1):

                # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
                train_data = self._dr_dataset.generateDataset()
                val_data = self._dr_dataset.generateDataset()

                # train and val loss lists
                train_loss_list = list()
                val_loss_list = list()
                train_mse_list = list()
                val_mse_list = list()
                
                # loop over trials
                for x_train, y_train, mask_train, x_val, y_val, mask_val in zip(train_data['x'],train_data['y'],train_data['mask'],
                                                                                val_data['x'],val_data['y'],val_data['mask']):

                    # plt.figure(figsize=(18,12))

                    # plt.subplot(231)
                    # plt.plot(x_train[:,0],label='x0',marker=None)
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(232)
                    # plt.plot(x_train[:,1],label='x1',marker=None)
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(233)
                    # plt.scatter(x_train[:,0],x_train[:,1],label='x',marker=None)
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(234)
                    # plt.plot(x_val[:,0],label='x0',marker=None)
                    # plt.grid()
                    # plt.legend()

                    # plt.subplot(235)
                    # plt.plot(x_val[:,1],label='x1',marker=None)
                    # plt.grid()

                    # plt.legend()

                    # plt.subplot(236)
                    # plt.scatter(x_val[:,0],x_val[:,1],label='x',marker=None)
                    # plt.grid()
                    # plt.legend()
                    
                    # plt.show()
                    # plt.close()
                    
                    # generate minibatches
                    x_train_mb, y_train_mb, mask_train_mb = self._generateMinibatches(x_train,y_train,mask_train)

                    # Generate LSTM 3-rank tensors
                    x_train_mb, y_train_mb, mask_train_mb = self._generateDataset(x_train_mb, y_train_mb, mask_train_mb) if self._train_ground else self._generateDataset(x_train_mb, x_train_mb, mask_train_mb)
                    x_val, y_val, mask_val = self._generateDataset(np.expand_dims(x_val,axis=0), np.expand_dims(y_val,axis=0), np.expand_dims(mask_val,axis=0)) if self._train_ground else \
                        self._generateDataset(np.expand_dims(x_val,axis=0), np.expand_dims(x_val,axis=0), np.expand_dims(mask_val,axis=0))

                    for x_mb, y_mb, mask_mb in zip(x_train_mb,y_train_mb,mask_train_mb):

                        # mask_mb = np.ones(shape=mask_mb.shape)
                        
                        train_feed_dict.update({self._X:x_mb,self._y:y_mb,self._mask:mask_mb})
                        sess.run(self._optimizer_op, feed_dict=train_feed_dict)


                    train_loss, train_mse = sess.run([self._loss_op,self._mse_op],feed_dict=train_feed_dict)
                    train_loss_list.append(train_loss)
                    train_mse_list.append(train_mse)
                    
                    for x_v, y_v, mask_v in zip(x_val,y_val,mask_val):
                    
                        val_feed_dict.update({self._X:x_v,self._y:y_v,self._mask:mask_v})
                        val_loss, val_mse = sess.run([self._loss_op,self._mse_op],feed_dict=val_feed_dict)
                        val_loss_list.append(val_loss)
                        val_mse_list.append(val_mse)

                    self._history['train_loss'].append(np.asarray(train_loss).mean())
                    self._history['val_loss'].append(np.asarray(val_loss).mean())
                    self._history['train_mse'].append(np.asarray(train_mse).mean())
                    self._history['val_mse'].append(np.asarray(val_mse).mean())

                    print('Epoch {epoch}, Training Loss/MSE {train_loss:0.4}/{train_mse:0.4}, Val Loss/MSE {val_loss:0.4}/{val_mse:0.4}'.format(epoch=epoch,
                                                                                                                                                train_loss=self._history['train_loss'][-1],
                                                                                                                                                train_mse=self._history['train_mse'][-1],        
                                                                                                                                                val_loss=self._history['val_loss'][-1],
                                                                                                                                                val_mse=self._history['val_mse'][-1]))
                    
            self._history['runtime'] = (time() - start_time) / 60.0

            if save_model:

                self._saveModel(sess,'trained_model.ckpt')

        return self._history
    
    def _buildNetwork(self):

        self._setPlaceholders()

        # weight regularizer
        try:

            self._weight_regularizer = self._weight_regularizer(weight_regularizer_scale)

        except:

            self._weight_regularizer = None

        self._y_hat = self._X
            
        for layer in range(len(self._hidden_dims)):

            input_timesteps = (self._seq_len) if layer == 0 else None
            input_dim = self._input_dim if layer == 0 else None
            return_seq = self._return_seq if layer < len(self._hidden_dims)-1 else False

            print('Input timesteps: {input_timesteps}'.format(input_timesteps = input_timesteps))
            print('Input Dim: {input_dim}'.format(input_dim = input_dim))
            print('Return Seq: {return_seq}'.format(return_seq=return_seq))
            print('units: {units}'.format(units = self._hidden_dims[layer]))

            if layer == 0 and self._stateful:

                # different inputs to first layer due to stateful parameter
                self._y_hat = LSTM_LAYER(
                                     units = self._hidden_dims[layer],
                                     activation = self._activation,
                                     batch_input_shape = (self._seq_len, input_timesteps, input_dim),
                                     bias_initializer = initializers.Constant(value=self._bias_initializer),
                                     kernel_regularizer = self._weight_regularizer,
                                     recurrent_regularizer = self._recurrent_regularizer,
                                     kernel_constraint = self._weight_constraint,
                                     return_sequences = return_seq,
                                     stateful = self._stateful,
                                     dropout=self._input_dropout_rate)(self._y_hat)

            else:

                # different inputs to first layer due to stateful parameter
                self._y_hat = LSTM_LAYER(
                                     units = self._hidden_dims[layer],
                                     activation = self._activation,
                                     input_shape = (input_timesteps, input_dim),
                                     bias_initializer = initializers.Constant(value=self._bias_initializer),
                                     kernel_regularizer = self._weight_regularizer,
                                     recurrent_regularizer = self._recurrent_regularizer,
                                     kernel_constraint = self._weight_constraint,
                                     return_sequences = return_seq,
                                     stateful = self._stateful,
                                     dropout=self._dropout_rate)(self._y_hat)

        self._y_hat = Dense(units=self._output_dim)(self._y_hat)

    def _setPlaceholders(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float32, shape=(None,self._seq_len,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float32, shape=(None,self._input_dim), name='y')
        self._mask = tf.placeholder(dtype=tf.float32, shape=(None,self._input_dim), name='mask')

    def _generateDataset( self, x, y, mask=None ):

        x_wins = list()
        y_wins = list()
        mask_wins = list()
            
        #for trial_idx in range(x.shape[0]):
        for x_trial,y_trial in zip(x,y):

            x_wins_trial, y_wins_trial = list(), list()

            for sample_idx in range(x_trial.shape[0]-self._seq_len):

                x_wins_trial.append(x_trial[sample_idx:sample_idx+self._seq_len,:])
                y_wins_trial.append(y_trial[sample_idx+self._seq_len,:])

            x_wins.append(np.asarray(x_wins_trial))
            y_wins.append(np.asarray(y_wins_trial))

        # generate mask
        if mask is not None:
            
            for mask_trial in mask:

                mask_wins_trial = list()

                for sample_idx in range(x_trial.shape[0]-self._seq_len):

                    mask_wins_trial.append(mask_trial[sample_idx+self._seq_len,:])

                mask_wins.append(np.asarray(mask_wins_trial))
            
        return x_wins, y_wins, mask_wins
