import os, errno
import tensorflow as tf
import numpy as np
import random
import copy
from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict, loadDict, generateMask
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import shapiro
import copy

class DomainRandomizationDataset(AbstractDataset):

    def __init__(self, params=None):

        super().__init__(params)

    ##################
    # Public Methods #
    ##################

    def getDataset(self, load_path):

        assert load_path is not None

        # load dataset using load_path attribute
        self._data = loadDict(load_path)

        return self._data

    def generateDataset(self):

        """
        Generate domain randomization dataset.  Logic for whether to save to disk or return in in getDataset()
        """

        # linspace for generating function values
        t = np.expand_dims(np.linspace(self._x_range[0], self._x_range[1], self._n_samples), axis=-1)
                
        # list to hold either training or testing datasets.
        self._data['x_test'] = list()
        self._data['y_test'] = list()
        noise_types = list()

        for trial in range(self._n_trials):

            # randomly select training function and parameters
            self._fn_name, self._fn_def, self._fn_params = random.choice(self._fns)

            y_loop_list = list()

            for _ in range(self._n_features):

                param_list = list()

                for param in self._fn_params:

                    if isinstance(param, tuple):

                        param_list.append(np.random.uniform(param[0], param[1]))

                    else:

                        param_list.append(param)

                # select the number of parameters if polynomial order is randomized
                try:

                    param_max_index = np.random.randint(self._min_N,self._max_N+1) + 1
                    param_list = param_list[:param_max_index]

                except:

                    pass

                y_loop_list.append(np.concatenate([np.zeros((self._n_baseline_samples,1)),self._fn_def(t, param_list)]))

            y = np.hstack(y_loop_list)

            ###############################################
            # randomly select training noise and parameters
            ###############################################

            self._noise_name, self._noise_dist, self._noise_params = random.choice(self._noise)
            noise_types.append(self._noise_name)

            if self._noise_name is not None:
            
                # randomly select noise params if tuple
                noise_param_dict = dict()

                for param_key, param in self._noise_params.items():

                    if isinstance(param, tuple):

                        noise_param_dict[param_key] = np.random.uniform(param[0], param[1])

                    else:

                        noise_param_dict[param_key] = param

                noise = self._noise_dist(**noise_param_dict, size=(self._n_baseline_samples+self._n_samples,self._n_noise_features))

                x = y + noise

            else:

                x = y
                
            ######################
            # Shift y
            ######################

            # try:

            #     #shift_value = np.random.uniform(low=self._baseline_shift[0],high=self._baseline_shift[1],size=1)
            #     shift_value = np.random.uniform(high=scale_max_value)
            #     self._data['y_train'] += shift_value 
            #     self._data['y_val'] += shift_value
            #     self._data['y_test'] += shift_value
            #     self._data['x_train'] += shift_value
            #     self._data['x_val'] += shift_value
            #     self._data['x_test'] += shift_value

            # except:

            #     pass

            #####################
            # Add first dimension
            #####################

            self._data['y_test'].append(y)
            self._data['x_test'].append(x)

        # set dataset_dict
        self._data['y_test'] = np.asarray(self._data['y_test'])
        self._data['x_test'] = np.asarray(self._data['x_test'])
        self._data['t'] = np.expand_dims(np.linspace(self._x_range[0], self._x_range[1],
                                                     self._n_baseline_samples+self._n_samples), axis=-1)
        self._data['noise_type'] = noise_types
        self._data['mask'] = np.ones(shape=self._data['x_test'].shape)

        # add missing values
        if self._missing_percent != 0.0:

            for index, x in enumerate(self._data['x']):
                
                    mask_indices = generateMask(x,
                                                self._missing_percent/100.0)

                    self._data['x'][index][mask_indices] = self._missing_value
                    self._data['y'][index][mask_indices] = self._missing_value

            # if excluding missing values from loss function
            if self._with_mask:

                self._data['mask'][index][mask_indices] = 0.0

        # save dataset logic
        if getattr(self, '_save_path', None) is not None:

            try:

                os.makedirs(os.path.dirname(self._save_path))

            except OSError as e:

                if e.errno != errno.EEXIST:

                    raise

            saveAttrDict(save_dict=self.__dict__, save_path=self._save_path)

        else:

            return copy.copy(self._data)
