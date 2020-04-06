import os
import glob
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict, generateMask

import matplotlib.pyplot as plt

class padsDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        datadir: path to text file directory
        datafile: name of text file
        activity: activity to use for dataset
        min_len: minimum length of time series to include in dataset
        max_len: break datasets into subsets of this length
        standardize: whether to use StandardScaler, bool
        begin_pad: number of leading terms to exclude from possible missing data
        noise: noise type
        mask_percent: percentage of values in each time series to mask
        mask_value: fill in for missing values

        """

        super().__init__(params)

        # map tag id to text
        self._tag_dict = {
                '010-000-024-033':'ANKLE_LEFT',
                '020-000-033-111':'CHEST',
                '020-000-032-221':'BELT',
                '010-000-030-096':'ANKLE_RIGHT',
            }
        
    ##################
    # Public Methods #
    ##################

    def getDataset(self,np_filename=None):

        # load previously saved dataset
        if hasattr(self,'_saved_dataset'):

            self._data = self.getSavedDataset(self._saved_dataset)

        # create and save dataset
        else:

            ###################################
            # Read raw data
            ###################################
            raw_data = pd.read_csv(self._datadir+'raw/'+self._datafile)

            ###################################
            # Read raw data into dictionary
            ###################################
            # list of dictionaries, one for each trial
            coords_dict = dict()

            for seq_name in raw_data.sequence_name.unique()[:]:

                for tag in raw_data.tag_id.unique()[:]:

                    for activity in raw_data.activity.unique()[:]:

                        if activity == self._activity:

                            data = raw_data[ (raw_data.sequence_name==seq_name) & (raw_data.tag_id==tag) & (raw_data.activity==activity) ]

                            coords = np.hstack([
                                np.expand_dims(data['x'].values,axis=-1),
                                np.expand_dims(data['y'].values,axis=-1),
                                np.expand_dims(data['z'].values,axis=-1)
                            ])

                            n_splits = coords.shape[0]//self._max_len

                            x0, x1 = 0, self._max_len

                            key_index = 1
                            
                            for split in np.arange(n_splits):

                                #print(coords[x0:x1].shape)
                                x0 = x1
                                x1 += self._max_len

                                if coords[x0:x1].shape[0] > self._min_len:

                                    coords_dict_key = ('_').join([seq_name,self._tag_dict[tag],activity,'{key_index}'.format(key_index=key_index)])
                                    coords_dict[coords_dict_key] = coords[x0:x1]
                                    key_index += 1

                            if coords[x0:].shape[0] > self._min_len:

                                coords_dict_key = ('_').join([seq_name,self._tag_dict[tag],activity,'{key_index}'.format(key_index=key_index)])
                                coords_dict[coords_dict_key] = coords[x0:]

        ###################################
        # Standardize Data
        ###################################

        # stack all data for standardization                                                                                             
        for k,v in coords_dict.items():

            try:

                X = np.concatenate([X,v])

            except:

                X = v

        # standardize                                                                                                                    
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # Map standardized data to list
        coords_list = list()
        x0, x1 = 0,0
        for k,v in coords_dict.items():

            x1 = len(v) + x1
            coords_list.append(X_s[x0:x1])
            x0 = x1

        ################################################
        # Write to data dictionary and add noise to data
        ################################################

        self._data['x_true'] = np.asarray(coords_list)
        self._data['keys'] = list(coords_dict.keys())

        if self._noise[0] is not 'none':

            coords_list_noise = list()

            for trial in coords_list:

                coords_list_noise.append(trial + self._noise[1](**self._noise[2],size=trial.shape))

            self._data['x_test'] = np.asarray(coords_list_noise)

        else:

            self._data['x_test'] = copy.deepcopy(self._data['x_true'])

        ############
        # Apply mask
        ############

        if self._mask_percent is not 0.0:

            for index, x_test in enumerate(self._data['x_test']):

                mask_indices = generateMask(x_test,
                                            self._mask_percent/100.0,
                                            self._begin_pad)

                #self._data['x_test'][mask_indices] = self._mask_value
                self._data['x_test'][index][mask_indices] = self._mask_value

        ##############
        # save to disk
        ##############

        save_dir = self._datadir + 'split/' + self._dataset_name.replace(' ','_') + '.pkl'
        saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        ###################                
        return self._data
