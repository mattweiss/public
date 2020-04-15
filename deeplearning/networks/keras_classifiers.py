import os
from time import time
import numpy as np
from scipy import stats
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pdb import set_trace as st
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from dovebirdia.utilities.base import dictToAttributes, saveAttrDict, saveDict
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset

try:

    import matplotlib.pyplot as plt

except:

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

class MaxScale(tf.keras.layers.Layer):

  def __init__(self, units=None, input_dim=None):

    super(MaxScale, self).__init__()

  def call(self, inputs):

      return tf.divide(inputs,tf.expand_dims(tf.reduce_max(inputs,axis=1),axis=-1))

class KerasMultiLabelClassifier():

    """
    Sinlge label classifier in Keras
    """

    def __init__(self, params=None):

        assert isinstance(params,dict)

        dictToAttributes(self,params)

        # build model
        self._model = keras.Sequential()

        for dim_index, dim in enumerate(self._hidden_dims):

            input_dim = self._input_dim if dim_index == 0 else None

            if self._input_dropout_rate != 0:

                self._model.add(keras.layers.Dropout(self._input_dropout_rate, input_shape=(input_dim,)))

            self._model.add(keras.layers.Dense(dim,
                                               input_dim=input_dim,
                                               kernel_initializer = self._kernel_initializer,
                                               bias_initializer = self._bias_initializer,
                                               kernel_regularizer=self._kernel_regularizer(self._kernel_regularizer_scale),
                                               activation=self._activation))

            if self._dropout_rate != 0:

                self._model.add(keras.layers.Dropout(self._dropout_rate))

        # output layer
        self._model.add(keras.layers.Dense(self._output_dim,
                                           activation=self._output_activation))


        if self._scale_output:

            # scale output such that largest value is 1
            self._model.add(MaxScale(units=self._output_dim,input_dim=self._output_dim))

        # compile model

        if self._loss.__name__ == 'categorical_crossentropy':

            loss = [lambda y_true,y_pred: self._loss(y_true, y_pred, from_logits=self._from_logits)]

        elif self._loss.__name__ == 'sigmoid_cross_entropy_with_logits':

                loss = [lambda y_true,y_pred: self._loss(labels=y_true, logits=y_pred)]

        elif self._loss.__name__ == 'mean_squared_error':

                loss = self._loss

        self._model.compile(optimizer=self._optimizer(**self._optimizer_params),
                            loss=loss,
                            metrics=list(self._metrics)
                            )

        print(self._model.summary())

    def fit(self, dataset):

        # start time
        start_time = time()

        # define empty callbacks like
        self._callbacks_list = list()

        if self._early_stopping:

            self._callbacks_list.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50))

        history = self._model.fit(x=dataset['x_train'],
                                  y=dataset['y_train'][:,:self._output_dim],
                                  batch_size=self._mbsize,
                                  epochs=self._epochs,
                                  validation_data=(dataset['x_val'],dataset['y_val'][:,:self._output_dim]),
                                  callbacks=self._callbacks_list)

        # total runtime
        history.history['runtime'] = (time() - start_time) / 60.0

        # validataion set metrics
        # val_pred = self._model.predict(x=dataset['x_val'])
        # history.history['val_true'] = dataset['y_val'][:,:self._output_dim]
        # history.history['val_pred'] = val_pred
        # history.history['val_subset_accuracy'] = accuracy_score(y_true=history.history['val_true'],
        #                                                         y_pred=(history.history['val_pred'] >= 0.5).astype(float))
        # test set metrics
        #test_pred = self._model.predict(x=dataset['x_test'])
        #history.history['test_true'] = dataset['y_test'][:,:self._output_dim]
        #history.history['test_pred'] = test_pred
        # history.history['test_subset_accuracy'] = accuracy_score(y_true=history.history['test_true'],
        #                                                          y_pred=(history.history['test_pred'] >= 0.5).astype(float))

        # relabel training set metric keys
        for train_key in self._model.metrics_names:

            history.history['train_' + train_key] = history.history.pop(train_key)

        # add test set metrics to history
        test_metrics = [self._model.evaluate(x=dataset['x_test'],y=dataset['y_test'][:,:self._output_dim])]
        
        for metric_name,metric in zip(self._model.metrics_names,test_metrics[0]):

            metric_name = 'test_' + metric_name
            
            history.history[metric_name] = metric


        # dictionary with final returned results
        output_history = dict()
                
        # select last entry in each metrics list
        for k,v in history.history.items():
            
            # if list, i.e. training and validataion
            try:

                output_history[k] = history.history[k][-1]

            # if scalar, i.e. test metrics and runtime
            except:

                output_history[k] = history.history[k]
            
        return output_history
