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
        self._setOptimizer()

        ############################
        # Compile Model
        ############################
        # This is a fix since sometimes self._loss is passed as the function definition is config files are used
        try:
            
            self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)

        except:

            self._model.compile(optimizer=self._optimizer, loss=self._loss(), metrics=self._metrics)

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

        try:

            print(self._model.summary())
            tf.keras.utils.plot_model(self._model, 'my_first_model.png')

        except:

            print('Unable to print model summary')

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

        self._history = self._model.fit(self._x_train, self._y_train,
                                        batch_size=self._mbsize,
                                        epochs=self._epochs,
                                        validation_data=(self._x_val, self._y_val))

        if save_weights:

            self._model.save_weights('test.keras')

    def fitDomainRandomization(self, fns=None, save_weights=False):

        dictToAttributes(self,fns)

        # randomly select one of the functions in self._fns
        self._fn_name, self._fn_def, self._fn_params = random.choice(self._fns)

        # generate training and validation curves
        x_train, x_train_gt = self._generateDomainRandomizationData(self._fn_params)
        x_val, x_val_gt = self._generateDomainRandomizationData(self._fn_params)
            
        for epoch in range(1, self._epochs+1):

            # randomly select one of the functions in self._fns
            # self._fn_name, self._fn_def, self._fn_params = random.choice(self._fns)
        
            # # generate training and validation curves
            # x_train, x_train_gt = self._generateDomainRandomizationData(self._fn_params)
            # x_val, x_val_gt = self._generateDomainRandomizationData(self._fn_params)

            print('Epoch {epoch}'.format(epoch=epoch))

            history = self._model.fit(x_train, x_train,
                                      batch_size=100,
                                      epochs=1,
                                      validation_data=(x_val, x_val_gt),
                                      shuffle=False,
                                      verbose=2)

            self._history['train_loss'].append(history.history['loss'][0])
            self._history['val_loss'].append(history.history['val_loss'][0])
            
            # train_loss = self._model.train_on_batch(x_train, x_train_gt)
            # val_loss =  self._model.test_on_batch(x_val, x_val_gt)
            #print('Train Loss: {train_loss}, Val Loss: {val_loss}'.format(train_loss=train_loss, val_loss=val_loss))           
            # self._history['train_loss'].append(train_loss)
            # self._history['val_loss'].append(val_loss)

            if epoch == self._epochs:
            
                plt.figure(figsize=(12,6))
                plt.subplot(121)
                plt.plot(x_train, label='train')
                plt.plot(x_train_gt, label='train_gt')
                plt.plot(self._model.predict(x_train), label='train_pred')
                plt.grid()
                plt.legend()
                plt.subplot(122)
                plt.plot(x_val, label='val')
                plt.plot(x_val_gt, label='val_gt')
                plt.plot(self._model.predict(x_val), label='val_pred')
                plt.grid()
                plt.legend()
                plt.show()
                plt.close()
            
            # keep history lists to fixed length
            for loss_key in self._history.keys():

                if len(self._history[loss_key]) > self._test_size:

                    self._history[loss_key].pop()
            
        # plot test prediction
        for _ in range(self._test_size):

            x_test, x_test_gt = self._generateDomainRandomizationData(self._fn_params)
            self._history['test_loss'].append(self._model.evaluate(self._model.predict(x_test), x_test_gt, verbose=2))
                    
        if save_weights:

            self._model.save_weights('test.keras')

        print(self._model.metrics_names)
            
        return self._history
            
    def predict(self, dataset=None):

        pass
        
    def evaluate(self, dataset=None):

        pass
    
    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        # set input
        input = tf.keras.Input(shape=(self._input_dim,))

        output = self._buildDenseLayers(input, self._hidden_dims)(input)

        # output layer
        output = tf.keras.layers.Dense(units=self._output_dim,
                                       activation=self._output_activation,
                                       use_bias=self._use_bias,
                                       kernel_initializer=self._kernel_initializer,
                                       bias_initializer=self._bias_initializer,
                                       kernel_regularizer=self._kernel_regularizer,
                                       bias_regularizer=self._bias_regularizer,
                                       activity_regularizer=self._activity_regularizer,
                                       kernel_constraint=self._kernel_constraint,
                                       bias_constraint=self._bias_constraint)(output)

        self._model = tf.keras.Model(inputs=input, outputs=output)
        
    def _setOptimizer(self):

        if self._optimizer_name == 'adam':

            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        
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

        return tf.keras.Model(inputs=input, outputs=output, name=name)

    def _generateDomainRandomizationData(self, params):

        param_list = list()

        for param in params:

            if isinstance(param, tuple):

                param_list.append(np.random.uniform(param[0], param[1]))
                
            else:

                param_list.append(param)
                
        x = np.linspace(self._x_range[0], self._x_range[1], self._n_samples)
        x_gt = self._fn_def(x, *param_list)
        x = x_gt + self._noise(**self._noise_params, size=self._n_samples)

        return np.expand_dims(x, axis=-1), np.expand_dims(x_gt, axis=-1)
