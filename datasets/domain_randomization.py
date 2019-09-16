import tensorflow as tf
import numpy as np
import random
import copy
from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveDict

class DomainRandomizationDataset(AbstractDataset):

    def __init__(self, params=None):
    
        super().__init__(params)
    
    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        self._loadDataset()

        return self._dataset_dict

    ###################
    # Private Methods #
    ###################

    def _loadDataset(self):

        """
        Generate domain randomization dataset.  Logic for whether to save to disk or return in in getDataset()
        """

        # linspace for generating function values
        x = np.expand_dims(np.linspace(self._x_range[0], self._x_range[1], self._n_samples), axis=-1)

        # list to hold either training of testing datasets.
        x_list = list()
        y_list = list()
        
        # If training include list for validation data
        if self._ds_type == 'train':

            x_val_list = list()
            y_val_list = list()
            n_datasets = 2

        else:

            n_datasets = 1

        for trial in range(self._n_trials):
        
            # randomly select training function and parameters
            self._fn_name, self._fn_def, self._fn_params = random.choice(self._fns)

            # loop twice if training to generate validation set
            for dataset_ctr in range(n_datasets):

                param_list = list()

                for param in self._fn_params:

                    if isinstance(param, tuple):

                        param_list.append(np.random.uniform(param[0], param[1]))

                    else:

                        param_list.append(param)

                y = self._fn_def(x, param_list)
                y -= y[0]

                y_noise = y + self._noise(**self._noise_params, size=(self._n_samples,1))

                if dataset_ctr == 0:

                    x_list.append(y_noise)
                    y_list.append(y)

                else:

                    x_val_list.append(y_noise)
                    y_val_list.append(y)

        # set dataset_dict
        if self._ds_type == 'train':
            
            self._dataset_dict['x_train'] = np.asarray(x_list)
            self._dataset_dict['y_train'] = np.asarray(y_list)
            self._dataset_dict['x_val'] = np.asarray(x_val_list)
            self._dataset_dict['y_val'] = np.asarray(y_val_list)

        else:

            self._dataset_dict['x_test'] = np.asarray(x_list)
            self._dataset_dict['y_test'] = np.asarray(y_list)

        # save dataset logic
        if getattr(self, '_save_path', None) is not None:

            saveDict(save_dict=self.__dict__, save_path=self._save_path)
            
            

            