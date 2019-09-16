import numpy as np
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset

class UNPAOPDataset(AbstractDataset):

    def __init__(self, params=None):
    
        super().__init__(params)

    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # run loadDataset
        self._loadDataset()
        
        # load data
        return self._dataset_dict

    ###################
    # Private Methods #
    ###################

    def _loadDataset(self):

        # read data from disk
        self._dataset_dir = self._dataset_base_dir + 'user_{user}_video_{video}'.format(user=self._user, video=self._video) + '/data/'
        self._dataset_file = 'unpaop_user_{user}_video_{video}.npy'.format(user=self._user, video=self._video)
        self._dataset = np.load(self._dataset_dir+self._dataset_file, allow_pickle=True).item()

        # landmarks to load
        self._landmark_names = self._landmarks.keys()

        # get UNPA data
        self._unpa_data = list()
        self._unpa_labels = list()
        self._op_data = list()
        self._op_labels = list()

        for lm, data in self._dataset.items():

            if lm.split('_')[-1] in self._landmark_names:

                if 'unpa' in lm.split('_')[0]:

                    self._unpa_data.append(data)
                    self._unpa_labels.append(lm)

                elif 'op' in lm.split('_')[0]:

                    self._op_data.append(data)
                    self._op_labels.append(lm)
                
        self._unpa_data = np.asarray(self._unpa_data)
        self._unpa_labels = np.asarray(self._unpa_labels)
        self._op_data = np.asarray(self._op_data)
        self._op_labels = np.asarray(self._op_labels)

        self._dataset_dict['x'] = self._unpa_data
        self._dataset_dict['y'] = self._unpa_labels
        self._dataset_dict['x_noise'] = self._op_data
        self._dataset_dict['y_noise'] = self._op_labels
