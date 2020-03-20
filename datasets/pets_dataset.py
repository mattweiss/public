import os
import glob
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict, generateMask

import matplotlib.pyplot as plt

class petsDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        datadir: path to xml file directory
        dataset: name of xml file
        samples: tuple of samples to include (start,end)
        standardize: whether to use StandardScaler, bool

        """

        super().__init__(params)
        
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
            # Read XML Data
            ###################################
            tree = ET.parse(self._datadir+'raw/'+self._dataset)
            root = tree.getroot()

            coords_dict = dict()

            for node in root: 

                for obj in node.find("objectlist"):
        
                    id = int(obj.attrib['id'])
                    xc = float(obj.find("box").attrib['xc'])
                    yc = float(obj.find("box").attrib['yc'])

                    if id not in coords_dict.keys():
                        
                        coords_dict[id] = list()
            
                    coords_dict[id].append([xc,yc])

            ###################################
            # Standardize Data
            ###################################

            if self._standardize:

                # stack all data for standardization
                for k,v in coords_dict.items():

                    try:

                        X = np.concatenate([X,v])

                    except:

                        X = v

                # standardize
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)

                # Map standardized data to new dictionary
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
                                                self._mask_percent)

                    #self._data['x_test'][mask_indices] = self._mask_value
                    self._data['x_test'][index][mask_indices] = self._mask_value

            ##############
            # save to disk
            ##############

            save_dir = self._datadir + 'split/' + self._dataset_name.replace(' ','_') + '.pkl'
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)
                    
        ###################                
        return self._data
