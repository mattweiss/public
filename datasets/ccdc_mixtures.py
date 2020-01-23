import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset

class ccdcMixturesDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        with_val - include validation set
        resistance_type - raw measurements, z scored, Kalman Filtered, etc.
        labels - analyte ratios to include
        sensors - sensors to include
        with_synthetic - include synthetic data
        sample_indices - indices to slice samples
        multi_label - boolean indicating multi label classification or not
        """

        super().__init__(params)

    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # read training and testing pickle files
        self._pickle_files_train = os.listdir(self._dataset_dir + 'training/')

        if self._with_val:

            self._pickle_files_val = os.listdir(self._dataset_dir + 'validation/')

        self._pickle_files_test = os.listdir(self._dataset_dir + 'testing/')

        # by default all files are read.  If not using synthetic remove here.
        if not self._with_synthetic:

            self._pickle_files_train = [ f for f in self._pickle_files_train if 'synthetic' not in f ]

        # read pickle files and generate pandas dataframes
        self._data_train = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/training/' + pf ) for pf in self._pickle_files_train ] )

        if self._with_val:

            self._data_val  = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/validation/' + pf ) for pf in self._pickle_files_val ] )

        self._data_test  = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/testing/' + pf ) for pf in self._pickle_files_test ] )

        # training set
        self._data['x_train'] = np.asarray([ trial[self._samples[0]:self._samples[1],:] for trial in self._data_train[self._resistance_type].values ])
        self._data['x_train'] = self._data['x_train'].reshape(-1,  self._data['x_train'].shape[1]*self._data['x_train'].shape[2] )

        if self._multi_label:

            # self._data['y_train'] = np.stack(self._data_train['concentration'].apply(lambda x : [ float(i) for i in x.split(",") ]))
            # self._data['y_train'] = np.where(self._data['y_train']!=0,1,self._data['y_train'])
            self._data['y_train'] = np.stack(self._data_train.multi_label)

        else:

            self._data['y_train'] = np.asarray([np.squeeze(np.asarray(label)) for label in self._data_train.label])

        if self._with_val:

            # validation set
            self._data['x_val'] = np.asarray([ trial[self._samples[0]:self._samples[1],:] for trial in self._data_val[self._resistance_type].values ])
            self._data['x_val'] = self._data['x_val'].reshape(-1,  self._data['x_val'].shape[1]*self._data['x_val'].shape[2] )

            if self._multi_label:

                # self._data['y_val'] = np.stack(self._data_val['concentration'].apply(lambda x : [ float(i) for i in x.split(",") ]))
                # self._data['y_val'] = np.where(self._data['y_val']!=0,1,self._data['y_val'])
                self._data['y_val'] = np.stack(self._data_val.multi_label.values)

            else:

                self._data['y_val'] = np.asarray([np.squeeze(np.asarray(label)) for label in self._data_val.label])

        # testing set
        self._data['x_test'] = np.asarray([ trial[self._samples[0]:self._samples[1],:] for trial in self._data_test[self._resistance_type].values ])
        self._data['x_test'] = self._data['x_test'].reshape(-1,  self._data['x_test'].shape[1]*self._data['x_test'].shape[2] )

        if self._multi_label:

            # self._data['y_test'] = np.stack(self._data_test['concentration'].apply(lambda x : [ float(i) for i in x.split(",") ]))
            # self._data['y_test'] = np.where(self._data['y_test']!=0,1,self._data['y_test'])
            self._data['y_test'] = np.stack(self._data_test.multi_label)

        else:

            self._data['y_test'] = np.asarray([np.squeeze(np.asarray(label)) for label in self._data_test.label])

        if self._feature_range is not None:

            min_max_scaler = MinMaxScaler(feature_range=self._feature_range).fit(self._data['x_train'])
            self._data['x_train'] = min_max_scaler.transform(self._data['x_train'])

            if self._with_val:

                self._data['x_val'] = min_max_scaler.transform(self._data['x_val'])

            self._data['x_test'] = min_max_scaler.transform(self._data['x_test'])

        return self._data
