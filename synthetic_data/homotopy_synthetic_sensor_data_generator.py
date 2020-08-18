import os, sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import dill
from pdb import set_trace as st
from synthetic_data.synthetic_sensor_data_generator import SyntheticSensorDataGenerator

class HomotopySyntheticSensorDataGenerator(SyntheticSensorDataGenerator):

    """
    Produce synthetic sensor data using homotopy interpolation between 2 curves in each label group
    """

    def __init__( self,
                  dataset_dir = None,
                  trials = None,
                  sensors = None,
                  labels = None,
                  n_synthetic_sensors_per_label = 1,
                  save_plots = False):

        super().__init__(dataset_dir = dataset_dir,
                         trials = trials,
                         sensors = sensors,
                         labels = labels,
                         n_synthetic_sensors_per_label = n_synthetic_sensors_per_label,
                         save_plots = save_plots )

        # slice of samples to use - HARDCODE FOR NOW
        self._sample_slice = (None,None)

        # sampled frequency
        self._sample_freq = 20.0
        
    def generate_samples(self,save=True):

        """
        """

        # list to hold synthetic sensors
        self._synthetic_sensor_list = list()
        
        # group by class label
        self._labels = [ label for label in self._data.label.unique() if label in self._labels ] if self._labels is not None else self._data.label.unique()

        # synthetic sensor counter.  Used for naming synthetic sensor data dill files
        synthetic_sensor_ctr = 1

        # loop over labels
        for label in self._labels:

            # trials for given label
            label_data = self._data[self._data.label==label]

            # names of sensors
            sensor_names = list(label_data.sensors.iloc[0])

            # update sensors dictionary with values corresponding to sensor index
            # initiall the values in self._sensors are -1 because not all sensors will necessarily be used
            # to ensure the used sensors are labeled accordingly the below code is needed
            for sensor in self._sensors.keys():

                self._sensors[sensor] = sensor_names.index(sensor)

            # resistance values for given label, 3 dimensional array (trials, samples, sensors)
            resistance = np.asarray(list(label_data.resistance_z.values))

            # some resistance arrays are not 3d
            if np.ndim(resistance) == 3:

                resistance = resistance[:,:self._n_max_samples,:]

            else:

                continue

            # select subset of data
            resistance = resistance[:,self._sample_slice[0]:self._sample_slice[1],:]

            # generate synthetic sensor data
            for _ in range(self._n_synthetic_sensors_per_label):

                # dictionary for synthetic sensor data
                ssd_dict = dict.fromkeys(self._keys)

                # populate dictionary values
                ssd_dict['synthetic'] = True
                ssd_dict['label'] = label
                ssd_dict['concentration'] = label_data.iloc[0].concentration
                ssd_dict['event_indices'] = label_data.iloc[0].event_indices
                ssd_dict['name'] = label_data.iloc[0].name
                ssd_dict['sensors'] = list(self._sensors.keys())
                ssd_dict['time'] = label_data.iloc[0].time[:self._n_max_samples]
                ssd_dict['label'] = label_data.iloc[0].label
                ssd_dict['binary_presence_label'] = label_data.iloc[0].binary_presence_label
                ssd_dict['concentration_label'] = label_data.iloc[0].concentration_label
                #ssd_dict['y_multi_label'] = label_data.iloc[0].y

                # synthetic curve domain
                t = np.linspace(0,resistance.shape[1],resistance.shape[1]) / int(self._sample_freq)

                # loop over sensors
                #for sensor, sensor_idx in self._sensors.items():

                raw_curve_indicies = np.random.randint(low=0,high=resistance.shape[0],size=2)

                y_raw_1, y_raw_2 = list(resistance[raw_curve_indicies])

                homotopy_parameter = np.random.uniform()

                y_raw_syn = (homotopy_parameter**1)*y_raw_1 + (1.0-homotopy_parameter**1)*y_raw_2

                ssd_dict['resistance_z'] = y_raw_syn

                # pickle file name
                ssd_dict['pkl_file_name'] = label_data.iloc[0]['csv_file'].split('_')[0].replace('&','-') + '_synthetic_label_{label}_{syn_ctr}'.format(label=label, syn_ctr=str(synthetic_sensor_ctr))
                
                self._synthetic_sensor_list.append(ssd_dict)

                synthetic_sensor_ctr += 1

        if save:

            # save synthetic data
            for synthetic_sensor_dict in self._synthetic_sensor_list:

                with open(self._results_dir + synthetic_sensor_dict['pkl_file_name'] + '.pkl', 'wb') as handle:

                    dill.dump(synthetic_sensor_dict, handle, protocol=dill.HIGHEST_PROTOCOL)

        else:
                
            return self._synthetic_sensor_list
            
                # save plots
                # if self._save_plots:

                #     syn_label_ctr = 0

                #     #for sensor_idx, sensor in enumerate(self._sensors.keys()):
                #     for sensor, sensor_idx in self._sensors.items():

                #         fig = plt.figure(figsize=(6,6))

                #         ax1 = fig.add_subplot(111)

                #         trial_list = label_data.csv_file.values

                #         for trial_resistance in range(resistance.shape[0]):

                #             trial = int(trial_list[trial_resistance].split('_')[-1])
                #             ax1.plot(t,resistance[trial_resistance,:,sensor_idx], label = trial )
                #             syn_label_ctr = 1

                #         ax1.plot(t, ssd_dict['resistance'][:,sensor_idx], label='Synthetic', color='black', zorder=10 )
                #         ax1.set_xlabel('Sample')
                #         ax1.set_ylabel('Standardized Resistance')
                #         ax1.set_title('Real Sensor Response with Synthetic Overlay\nLabel {label}, {sensor}'.format(label=label, sensor=sensor))
                #         ax1.legend()
                #         ax1.grid()

                #         #plt.show()
                #         plt.savefig(self._figure_dir + synthetic_data_filename + '_{sensor}'.format(sensor=sensor).replace(' ','_'))
                #         plt.close()
