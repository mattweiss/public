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

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    def compile(self):

        super().compile()

        assert self._loss_op is not None
        assert self._optimizer_op is not None

        # compile model
        self._model.compile(loss = self._loss_op, optimizer=self._optimizer_op, metrics=['mse'])

        print(self._model.summary())

    def evaluate(self, x=None, y=None,labels=None,
                 loss_key='test_loss',
                 save_results=True):

        assert x is not None
        assert y is not None

        y_hat_list = list()

        model_results_path = './results/keras_model.h5'
        #self._model.load_weights(model_results_path)

        custom_objects={'leaky_relu': tf.nn.leaky_relu,}
        self._model = load_model(model_results_path, custom_objects=custom_objects)
        print(self._model.summary())

        x_test, y_test = self._generateDataset(x,y)

        for X,Y in zip(x_test,y_test):

            test_loss, test_mse = self._model.evaluate(x=X, y=Y, batch_size=X.shape[0], verbose=0)
            self._history['test_loss'].append(test_loss)
            self._history['test_mse'].append(test_mse)
            y_hat_list.append(self._model.predict(x=X, batch_size=X.shape[0]))

        # save predictions
        if save_results is not None:

            test_results_dict = {
                'x':x,
                'y':y,
                'y_hat':np.asarray(y_hat_list),
                'seq_len':self._seq_len,
                'labels':labels,
                }

            print(test_results_dict.keys())
            
            saveDict(save_dict=test_results_dict, save_path='./results/' + save_results + '.pkl')
        
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

            # lists to hold epoch training and validation losses
            epoch_train_loss = list()
            epoch_val_loss = list()
            epoch_train_mse = list()
            epoch_val_mse = list()

            # shuffle training set
            random_training_indices = np.random.choice(range(x_train.shape[0]), x_train.shape[0], replace=False)
            x_train, y_train = x_train[random_training_indices], y_train[random_training_indices]

            # loop over training examples
            for x_train_trial,y_train_trial in zip(x_train,y_train):

                # add outliers to training data
                if self._outliers:

                    train_outliers = generate_outliers(shape=y_train_trial.shape,
                                                       p_outlier=self._p_outlier,
                                                       outlier_range=self._outlier_range)
                    
                    # truth and truth + outliers
                    x_train_trial = x_train_trial + np.expand_dims(train_outliers,axis=-1) # expand dims since LSTM dataset is rank 3

                # plt.figure(figsize=(6,6))
                # plt.plot(x_train_trial[:,0,:],label='x',marker='o')
                # plt.plot(y_train_trial,label='y')
                # plt.grid()
                # plt.legend()
                # plt.title('Train')
                # plt.show()
                # plt.close()

                #############
                # train model
                #############
                
                # fit
                history = self._model.fit(x_train_trial,y_train_trial,
                                          batch_size=self._mbsize,
                                          verbose=0,
                                          epochs=1,
                                          shuffle=False)

                epoch_train_loss.append(history.history['loss'][0])
                epoch_train_mse.append(history.history['mse'][0])
                
                # evaluate on validation data
                for x_val_trial,y_val_trial in zip(x_val,y_val):
                        
                    # add outliers to validation data
                    if self._outliers:

                        val_outliers = generate_outliers(shape=x_val_trial.shape,
                                                         p_outlier=self._p_outlier,
                                                         outlier_range=self._outlier_range)
                    
                        # truth and truth + outliers
                        x_val_trial = x_val_trial + val_outliers

                    # evaluate validation curve
                    val_loss, val_mse = self._model.evaluate(x=x_val_trial, y=y_val_trial, batch_size=x_val_trial.shape[0], verbose=0)

                    epoch_val_loss.append(val_loss)
                    epoch_val_mse.append(val_mse)

                # plt.figure(figsize=(6,6))
                # plt.plot(x_val_trial[:,self._seq_len-1,:],label='x',marker='o')
                # plt.plot(y_val_trial,label='y')
                # plt.grid()
                # plt.title('Val')
                # plt.legend()
                # plt.show()
                # plt.close()
                    
            self._history['train_loss'].append(np.asarray(epoch_train_loss).mean())
            self._history['val_loss'].append(np.asarray(epoch_val_loss).mean())
            self._history['train_mse'].append(np.asarray(epoch_train_mse).mean())
            self._history['val_mse'].append(np.asarray(epoch_val_mse).mean())
            
            if epoch % 1 == 0:

                print('Epoch {epoch}, Training Loss {train_loss:0.4}, Val Loss {val_loss:0.4}'.format(epoch=epoch,
                                                                                                      train_loss=self._history['train_loss'][-1],
                                                                                                      val_loss=self._history['val_loss'][-1]))

        self._history['runtime'] = (time() - start_time) / 60.0

        # if test set was passed compute test loss
        try:

            epoch_test_loss = list()
            epoch_test_mse = list()

            # evaluate on validation data
            for x_test_trial,y_test_trial in zip(x_test,y_test):
            
                test_loss, test_mse = self._model.evaluate(x=x_test_trial, y=y_test_trial, batch_size=x_test_trial.shape[0], verbose=0)

                epoch_test_loss.append(test_loss)
                epoch_test_mse.append(test_mse)

            self._history['test_loss'].append(np.asarray(epoch_test_loss).mean())
            self._history['test_mse'].append(np.asarray(epoch_test_mse).mean())
  
        except:

            pass
        
        plt.figure(figsize=(6,6))
        plt.plot(self._history['train_loss'],label='Train Loss')
        plt.plot(self._history['val_loss'],label='Val Loss')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.savefig('loss_plot')

        if save_model:

            self._saveModel()

        return self._history

    def _fitDomainRandomization(self, dr_params=None, save_model=False):

        # create domainRandomizationDataset object
        self._dr_dataset = DomainRandomizationDataset(dr_params)

        start_time = time()

        with tf.Session() as sess:

            # initialize variables
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(1, self._epochs+1):

                # set x_train, y_train, x_val and y_val in dataset_dict attribute of DomainRandomizationDataset
                train_data = self._dr_dataset.generateDataset()
                val_data = self._dr_dataset.generateDataset()

                # loop over trials
                for x_train, y_train, x_val, y_val in zip(train_data['x'],train_data['y'],
                                                          val_data['x'],val_data['y']):

                    # train on minibatches
                    x_train_mb, y_train_mb = self._generateMinibatches(x_train,y_train)

                    # Generate LSTM 3-rank tensors
                    x_train_mb, y_train_mb = self._generateDataset(x_train_mb, y_train_mb) if self._train_ground else self._generateDataset(x_train_mb, x_train_mb)
                    x_val, y_val = self._generateDataset(np.expand_dims(x_val,axis=0), np.expand_dims(y_val,axis=0)) if self._train_ground else \
                        self._generateDataset(np.expand_dims(x_val,axis=0), np.expand_dims(x_val,axis=0))

                    for x_mb, y_mb in zip(x_train_mb,y_train_mb):

                        history = sess.run([self._model.fit(x_mb, y_mb,
                                                            batch_size=self._mbsize-self._seq_len,
                                                            verbose=2,
                                                            epochs=1)])
                        
                        # history = self._model.fit(x_mb, y_mb,
                        #                           batch_size=self._mbsize-self._seq_len,
                        #                           verbose=2,
                        #                           epochs=1,
                        #                           #validation_data=(x_v, y_v)
                        #)

                    val_loss, val_mse = self._model.evaluate(x=x_v, y=y_v, batch_size=self._mbsize-self._seq_len, verbose=0)
                    print(val_loss,val_mse)

                    self._history['train_loss'].append(history.history['loss'][0])
                    self._history['val_loss'] = val_loss #.append(history.history['val_loss'][0])
                    self._history['train_mse'].append(history.history['mse'][0])
                    self._history['val_mse'] = val_mse #.append(history.history['val_mse'][0])

        self._history['runtime'] = (time() - start_time) / 60.0

        if save_model:

            self._saveModel()

        return self._history

    def _buildNetwork(self):

        self._model = Sequential()

        # weight regularizer
        try:

            self._weight_regularizer = self._weight_regularizer(weight_regularizer_scale)

        except:

            self._weight_regularizer = None
            
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
                                     kernel_constraint = self._weight_constraint,
                                     return_sequences = return_seq,
                                     stateful = self._stateful,
                                     dropout=self._input_dropout_rate))

            else:

                # different inputs to first layer due to stateful parameter
                self._model.add(LSTM_LAYER(
                                     units = self._hidden_dims[layer],
                                     activation = self._activation,
                                     input_shape = (input_timesteps, input_dim),
                                     bias_initializer = initializers.Constant(value=self._bias_initializer),
                                     kernel_regularizer = self._weight_regularizer,
                                     recurrent_regularizer = self._recurrent_regularizer,
                                     kernel_constraint = self._weight_constraint,
                                     return_sequences = return_seq,
                                     stateful = self._stateful,
                                     dropout=self._dropout_rate))

        self._model.add(Dense(units=self._output_dim))

    def _setLoss(self):

        self._loss_op = self._loss(y_true,y_pred)
    
    def _setOptimizer(self):

        if self._optimizer.__name__ == 'Adam':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate)

        elif self._optimizer.__name__ == 'SGD':

            self._optimizer_op = self._optimizer(learning_rate=self._learning_rate,momentum=self._momentum,nesterov=self._use_nesterov)

    def _saveModel(self):

        self._trained_model_file = os.getcwd() + self._results_dir + 'keras_model.h5'
        #self._model.save_weights(self._trained_model_file)
        self._model.save(self._trained_model_file)

    def _generateDataset( self, x, y):

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
            
        return x_wins, y_wins
