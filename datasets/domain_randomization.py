import os, errno
import tensorflow as tf
import numpy as np
import random
import copy
from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict, loadDict
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import shapiro

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

        # list to hold either training of testing datasets.
        x_list = list()
        y_list = list()
        noise_types = list()

        # If training include list for validation data
        if self._ds_type == 'train':

            x_val_list = list()
            y_val_list = list()
            n_datasets = 2

        else:

            n_datasets = 1

        for trial in range(self._n_trials):

            # loop twice if training to generate validation set
            for dataset_ctr in range(n_datasets):

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

                # min max scale y
                try:

                    y = MinMaxScaler(feature_range=self._feature_range).fit_transform(y)

                except:

                    pass

                # zero curves
                y -= y[0]

                # if number of features is less than number of noise features
                if self._n_features < self._n_noise_features:

                    y = np.tile(y,(1,self._n_noise_features))

                # randomly select training noise and parameters
                self._noise_name, self._noise_dist, self._noise_params = random.choice(self._noise)
                noise_types.append(self._noise_name)

                # randomly select noise params if tuple
                noise_param_dict = dict()
                for param_key, param in self._noise_params.items():

                        if isinstance(param, tuple):

                            noise_param_dict[param_key] = np.random.uniform(param[0], param[1])

                        else:

                            noise_param_dict[param_key] = param

                y_noise = y + self._noise_dist(**noise_param_dict, size=(self._n_baseline_samples+self._n_samples,self._n_noise_features))

                # shift
                try:

                    shift_value = np.random.uniform(low=self._baseline_shift[0],high=self._baseline_shift[1],size=1)
                    y += shift_value
                    y_noise += shift_value

                except:

                    pass

                if dataset_ctr == 0:

                    x_list.append(y_noise)
                    y_list.append(y)

                else:

                    x_val_list.append(y_noise)
                    y_val_list.append(y)

        # set dataset_dict
        self._data['t'] = np.expand_dims(np.linspace(self._x_range[0], self._x_range[1], self._n_baseline_samples+self._n_samples), axis=-1)
        self._data['noise_type'] = noise_types

        if self._ds_type == 'train':

            self._data['x_train'] = np.asarray(x_list)
            self._data['y_train'] = np.asarray(y_list)
            self._data['x_val'] = np.asarray(x_val_list)
            self._data['y_val'] = np.asarray(y_val_list)

        else:

            self._data['x_test'] = np.asarray(x_list)
            self._data['y_test'] = np.asarray(y_list)

        # save dataset logic
        if getattr(self, '_save_path', None) is not None:

            try:

                os.makedirs(os.path.dirname(self._save_path))

            except OSError as e:

                if e.errno != errno.EEXIST:

                    raise

            saveAttrDict(save_dict=self.__dict__, save_path=self._save_path)

        else:

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(12,12))
            #
            # for feature_index in range(y_noise.shape[-1]):
            #
            #     plt.subplot(int('22{feature_index}'.format(feature_index=feature_index+1)))
            #
            #     plt.plot(self._data['y_train'][0,:,feature_index])
            #     plt.plot(self._data['x_train'][0,:,feature_index])
            #     plt.grid()
            #
            # plt.show()
            # plt.close()

            return self._data
