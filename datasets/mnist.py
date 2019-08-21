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

        # load data
        return self._loadDataset()

    ###################
    # Private Methods #
    ###################

    def _loadDataset(self):

        # Load from Keras
        (self._dataset_dict['x_train'], self._dataset_dict['y_train']), (self._dataset_dict['x_test'], self._dataset_dict['y_test']) = tf.keras.datasets.mnist.load_data()

        # preprocess
        self._dataset_dict['x_train'] = self._dataset_dict['x_train'].reshape(60000, 784).astype('float32') / 255
        self._dataset_dict['x_test'] = self._dataset_dict['x_test'].reshape(10000, 784).astype('float32') / 255

        # labels
        self._dataset_dict['y_train'] = self._dataset_dict['y_train'].astype('float32')
        self._dataset_dict['y_test'] = self._dataset_dict['y_test'].astype('float32')

        if self._with_val:
        
            # Reserve 10,000 samples for validation
            self._dataset_dict['x_val'] = self._dataset_dict['x_train'][-10000:]
            self._dataset_dict['y_val'] = self._dataset_dict['y_train'][-10000:]
            self._dataset_dict['x_train'] = self._dataset_dict['x_train'][:-10000]
            self._dataset_dict['y_train'] = self._dataset_dict['y_train'][:-10000]

        return self._dataset_dict
