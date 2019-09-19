import os
from time import time
import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM as LSTM_LAYER
from keras.models import Sequential
from keras import optimizers, regularizers, initializers, losses

from dovebirdia.deeplearning.networks.base import AbstractNetwork
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset

import matplotlib.pyplot as plt

class LSTM(AbstractNetwork):

    """
    LSTM Class
    """

    def __init__(self, params=None):

        super().__init__(params=params)

        # compile model
        self._model.compile( loss = self._loss_op, optimizer=self._optimizer_op )

        print(self._model.summary())
        
    ###################
    # Private Methods #
    ###################

    def fit(self, dataset=None, save_model=False):

        pass

    def fitDomainRandomization(self, dr_params=None, save_model=False):

        # create domainRandomizationDataset object
        self._dr_dataset = DomainRandomizationDataset(dr_params)

        start_time = time()
        
        for epoch in range(1, self._epochs+1):

            # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
            dr_data = self._dr_dataset.generateDataset()

            x_train, y_train = self._generate_dataset(dr_data['x_train'],dr_data['y_train'])
            x_val, y_val = self._generate_dataset(dr_data['x_val'],dr_data['y_val'])

            history = self._model.fit(x_train, y_train,
                                      batch_size=x_train.shape[0]-self._seq_len,
                                      epochs=1,
                                      validation_data=(x_val, y_val))

            self._history['train_loss'].append(history.history['loss'][0])
            self._history['val_loss'].append(history.history['val_loss'][0])

            if len(self._history['train_loss']) > self._history_size:

                self._history['train_loss'].pop(0)
                self._history['val_loss'].pop(0)
                
            # if epoch == self._epochs:

            #     train_pred = self._model.predict(x_train, batch_size = self._mbsize-self._seq_len)
            #     val_pred = self._model.predict(x_val, batch_size = self._mbsize-self._seq_len)

            #     plt.figure(figsize=(12,6))
            #     plt.subplot(121)
            #     plt.scatter(range(x_train.shape[0]), x_train[:,-1,:], label='train', color='green')
            #     plt.plot(y_train, label='train_gt')
            #     plt.plot(train_pred, label='train_pred')
            #     plt.grid()
            #     plt.legend()
            #     plt.subplot(122)
            #     plt.scatter(range(x_val.shape[0]), x_val[:,-1,:], label='val', color='green')
            #     plt.plot(y_val, label='val_gt')
            #     plt.plot(val_pred, label='val_pred')
            #     plt.grid()
            #     plt.legend()
            #     plt.show()
            #     plt.close()

            self._history['runtime'] = (time() - start_time) / 60.0
            
            if save_model:

                pass

        return self._history
                
    def predict(self, x=None):

        pass
        
    def evaluate(self, x=None, y=None, t=None, save_results=True):

        pass
    
    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        self._model = Sequential()

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
                self._model.add(LSTM_LAYER(
                                     units = self._hidden_dims[layer],
                                     activation = self._activation,
                                     #batch_input_shape=(self._mbsize-self._seq_len,self._seq_len,self._input_dim),
                                     batch_input_shape = (self._seq_len, input_timesteps, input_dim),
                                     bias_initializer = initializers.Constant(value=self._bias_initializer),
                                     kernel_regularizer = self._kernel_regularizer,
                                     recurrent_regularizer = self._recurrent_regularizer,
                                     return_sequences = return_seq,
                                     stateful = self._stateful))

            else:

                # different inputs to first layer due to stateful parameter
                self._model.add(LSTM_LAYER(
                                     units = self._hidden_dims[layer],
                                     activation = self._activation,
                                     input_shape = (input_timesteps, input_dim),
                                     bias_initializer = initializers.Constant(value=self._bias_initializer),
                                     kernel_regularizer = self._kernel_regularizer,
                                     recurrent_regularizer = self._recurrent_regularizer,
                                     return_sequences = return_seq,
                                     stateful = self._stateful))

        self._model.add(Dense(units=self._output_dim))
        
    def _setLoss(self):

        self._loss_op = self._loss
    
    def _setOptimizer(self):

        if self._optimizer.__name__ == 'Adam':

            self._optimizer_op = self._optimizer(lr=self._learning_rate)

    def _saveModel(self):

        pass

    def _generate_dataset( self, x, y ):

        x_wins = list()
        y_wins = list()

        for trial_idx in range( x.shape[0] ):

            for sample_idx in range( x.shape[1] - self._seq_len ):

                x_wins.append( x[ trial_idx, sample_idx:sample_idx + self._seq_len, : ] )
                y_wins.append( y[ trial_idx, sample_idx + self._seq_len, : ] )

        x_out = np.array( x_wins )
        y_out = np.array( y_wins )

        return x_out, y_out
