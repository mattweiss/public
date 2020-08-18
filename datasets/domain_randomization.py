import os, errno
import tensorflow as tf
import numpy as np
import random
import copy
from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict, loadDict
from scipy.stats import shapiro
import copy

class DomainRandomizationDataset(AbstractDataset):

    #def __init__(self, params=None):
    def __init__(self,
                 ds_type,
                 x_range,
                 n_trials,
                 n_baseline_samples,
                 n_samples,
                 n_features,
                 n_noise_features,
                 standardize,
                 feature_range,
                 baseline_shift,
                 param_range,
                 max_N,
                 min_N,
                 metric_sublen,
                 fns,
                 noise
                 ):

        self._ds_type=ds_type
        self._x_range=x_range
        self._n_trials=n_trials
        self._n_baseline_samples=n_baseline_samples
        self._n_samples=n_samples
        self._n_features=n_features
        self._n_noise_features=n_noise_features
        self._standardize=standardize
        self._feature_range=feature_range
        self._baseline_shift=baseline_shift
        self._param_range=param_range
        self._max_N=max_N
        self._min_N=min_N
        self._metric_sublen=metric_sublen
        self._fns=fns
        self._noise=noise

        # Attributes with default values
        self._data = dict()

        #super().__init__(params)

    ##################
    # Public Methods #
    ##################

    def getDataset(self, load_path=None):

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

                        #param_list.append(np.random.uniform(param[0], param[1]))
                        param_list.append(np.random.uniform(param[1]/2, param[1]))
                        param_list[-1] *= np.random.choice([-1,1])
                        
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

            y -= y[0,:]
            
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

                # if Bimodal
                if self._noise_name == 'bimodal':

                    noise = self._noise_dist(**noise_param_dict, size=(self._n_baseline_samples+self._n_samples))

                # if Gaussian or Cauchy
                else:

                    noise = self._noise_dist(**noise_param_dict, size=(self._n_baseline_samples+self._n_samples,self._n_features))

                x = y + noise

            else:

                x = y

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
