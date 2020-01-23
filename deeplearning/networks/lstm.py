import os, socket
from time import time
import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM as LSTM_LAYER
from keras.models import Sequential, load_model
from keras import optimizers, regularizers, initializers, losses

from dovebirdia.deeplearning.networks.base import AbstractNetwork, FeedForwardNetwork
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset

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

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    def compile(self):

        super().compile()

        assert self._loss_op is not None
        assert self._optimizer_op is not None

        # compile model
        self._model.compile(loss = self._loss_op, optimizer=self._optimizer_op)

        print(self._model.summary())

    def evaluate(self, x=None, y=None,
                 #t=None,
                 save_results=True):

        assert x is not None
        assert y is not None
        #assert t is not None

        x_hat_list = list()

        model_results_path = './results/keras_model.h5'
        #self._model.load_weights(model_results_path)

        custom_objects={'leaky_relu': tf.nn.leaky_relu,}
        self._model = load_model(model_results_path, custom_objects=custom_objects)
        print(self._model.summary())

        x_test, y_test = self._generateDataset(x,y)

        X_list = list()

        for X,Y in zip(x_test,y_test):

            self._history['test_loss'].append(self._model.evaluate(x=X, y=Y, batch_size=X.shape[0], verbose=0))
            x_hat_list.append(self._model.predict(x=X, batch_size=X.shape[0]))

        x_hat = np.asarray(x_hat_list)

        # save predictions
        if save_results:

            test_results_dict = {
                'x':x,
                'y':y,
                'x_hat':x_hat,
                # 't':t,
                }

        saveDict(save_dict=test_results_dict, save_path='./results/testing_results.pkl')

        return self._history

    def predict(self, x=None):

        pass

    ###################
    # Private Methods #
    ###################

    def _fit(self, dataset=None, save_model=False):

        x_train, y_train = self._generateDataset(dataset['x_train'],dataset['x_train'])
        x_val, y_val = self._generateDataset(dataset['x_val'],dataset['x_val'])
        x_test, y_test = self._generateDataset(dataset['x_test'],dataset['x_test'])

        start_time = time()

        for epoch in range(1, self._epochs+1):

            epoch_train_loss = list()
            epoch_val_loss = list()
            test_loss = list()
            test_pred = list()

            # train model
            for train_index in range(x_train[self._securities].shape[0]):

                history = self._model.fit(x_train[train_index], y_train[train_index],
                                          batch_size=x_train.shape[0],
                                          verbose=0,
                                          epochs=1,
                                          shuffle=False,
                                          validation_data=(x_val[train_index], y_val[train_index])
                                          )

                epoch_train_loss.append(history.history['loss'][0])

            # validation set
            for val_index in range(x_val[self._securities].shape[0]):

                epoch_val_loss.append(self._model.evaluate(x_val[val_index], y_val[val_index],verbose=0))

            self._history['train_loss'].append(np.asarray(epoch_train_loss).mean())
            self._history['val_loss'].append(np.asarray(epoch_val_loss).mean())

            if epoch % 1 == 0:

                print('Epoch {epoch}, Training Loss {train_loss:0.4}, Val Loss {val_loss:0.4}'.format(epoch=epoch,
                                                                                                      train_loss=self._history['train_loss'][-1],
                                                                                                      val_loss=self._history['val_loss'][-1]))

        # testing set
        for x,y in zip(x_test[self._securities],y_test[self._securities]):

            test_loss.append(self._model.evaluate(x,y,verbose=0))
            test_pred.append(self._model.predict(x,verbose=0))

        self._history['test_loss'] = np.asarray(test_loss).mean()
        self._history['test_pred'] = np.asarray(test_pred)
        self._history['test_true'] = np.asarray(y_test[self._securities])

        self._history['runtime'] = (time() - start_time) / 60.0

        if save_model:

            self._saveModel()

        print('Testing Loss {test_loss:0.4}'.format(test_loss=self._history['test_loss']))

        return self._history

    def _fitDomainRandomization(self, dr_params=None, save_model=False):

        # create domainRandomizationDataset object
        self._dr_dataset = DomainRandomizationDataset(dr_params)

        start_time = time()

        for epoch in range(1, self._epochs+1):

            # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
            dr_data = self._dr_dataset.generateDataset()

            x_train, y_train = self._generateDataset(dr_data['x_train'],dr_data['y_train'])
            x_val, y_val = self._generateDataset(dr_data['x_val'],dr_data['y_val'])

            x_train = np.squeeze(x_train,axis=0)
            y_train = np.squeeze(y_train,axis=0)
            x_val = np.squeeze(x_val,axis=0)
            y_val = np.squeeze(y_val,axis=0)

            print('Epoch {epoch}'.format(epoch=epoch))

            history = self._model.fit(x_train, y_train,
                                      batch_size=x_train.shape[0],
                                      verbose=2,
                                      epochs=1,
                                      validation_data=(x_val, y_val))

            self._history['train_loss'].append(history.history['loss'][0])
            self._history['val_loss'].append(history.history['val_loss'][0])

            # if len(self._history['train_loss']) > self._history_size:

            #     self._history['train_loss'].pop(0)
            #     self._history['val_loss'].pop(0)

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

        # Run test set
        # test_dataset_path = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/evaluation/kalman_filter_F_H/FUNC_taylor_NOISE_bimodal_LOC_0-05_SCALE_0-1_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_3_7.pkl'
        # dr_params['ds_type'] = 'test'
        # dataset = DomainRandomizationDataset(dr_params).getDataset(test_dataset_path)
        # x_test, y_test = dataset['data']['x_test'], dataset['data']['y_test']
        # x_test, y_test = self._generateDataset(x_test, y_test)

        # for x,y in zip(x_test,y_test):
        #
        #     test_loss = self._model.evaluate(x,y,verbose=0)
        #     self._history['test_loss'].append(test_loss)

        self._history['runtime'] = (time() - start_time) / 60.0

        if save_model:

            self._saveModel()

        return self._history

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
                                     batch_input_shape = (self._seq_len, input_timesteps, input_dim),
                                     bias_initializer = initializers.Constant(value=self._bias_initializer),
                                     kernel_regularizer = self._weight_regularizer,
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
                                     kernel_regularizer = self._weight_regularizer,
                                     recurrent_regularizer = self._recurrent_regularizer,
                                     return_sequences = return_seq,
                                     stateful = self._stateful))

        self._model.add(Dense(units=self._output_dim))

    def _setLoss(self):

        self._loss_op = self._loss
    #
    def _setOptimizer(self):

        if self._optimizer.__name__ == 'Adam':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate)

        elif self._optimizer.__name__ == 'Momentum':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate,momentum=self._momentum,use_nesterov=self._use_nesterov)

    def _saveModel(self):

        self._trained_model_file = os.getcwd() + self._results_dir + 'keras_model.h5'
        #self._model.save_weights(self._trained_model_file)
        self._model.save(self._trained_model_file)

    def _generateDataset( self, x, y ):

        x_wins = list()
        y_wins = list()

        for trial_idx in range(x.shape[0]):

            for sample_idx in range(x.shape[1]-self._seq_len):

                x_wins.append(x[trial_idx,sample_idx:sample_idx+self._seq_len,:])
                y_wins.append(y[trial_idx,sample_idx+self._seq_len,:])

        x_out = np.array(x_wins).reshape((x.shape[0],-1,self._seq_len,x.shape[-1]))
        y_out = np.array(y_wins).reshape((x.shape[0],-1,x.shape[-1]))

        return x_out, y_out
