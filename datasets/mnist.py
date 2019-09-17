import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset

class MNISTDataset(AbstractDataset):

    def __init__(self, params=None):
    
        super().__init__(params)
    
    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # Load from Keras
        (self._data['x_train'], self._data['y_train']), (self._data['x_test'], self._data['y_test']) = tf.keras.datasets.mnist.load_data()

        # preprocess
        self._data['x_train'] = self._data['x_train'].reshape(60000, 784).astype('float64') / 255
        self._data['x_test'] = self._data['x_test'].reshape(10000, 784).astype('float64') / 255

        if self._supervised:
            
            # supervised
            self._data['y_train'] = self._data['y_train'].astype('float64')
            self._data['y_test'] = self._data['y_test'].astype('float64')

        else:

            # unsupervised
            self._data['y_train'] = self._data['x_train']
            self._data['y_test'] = self._data['x_test']
        
        if self._with_val:
        
            # Reserve 10,000 samples for validation
            self._data['x_train'] = self._data['x_train'][:-10000]
            self._data['x_val'] = self._data['x_train'][-10000:]

            # supervised
            if self._supervised:
            
                self._data['y_train'] = self._data['y_train'][:-10000]
                self._data['y_val'] = self._data['y_train'][-10000:]

            # unsupervised
            else:

                self._data['y_train'] = self._data['x_train']
                self._data['y_val'] = self._data['x_val']           
            
        if self._onehot:

            self._data['y_train'] = tf.keras.utils.to_categorical(self._data['y_train'], 10)
            self._data['y_val'] = tf.keras.utils.to_categorical(self._data['y_val'], 10)
            self._data['y_test'] = tf.keras.utils.to_categorical(self._data['y_test'], 10)
            
        return self._data
