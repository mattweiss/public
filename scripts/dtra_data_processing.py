#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:0

import pandas as pd
import numpy as np

# import matplotlib
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import argparse
import dill

import xlrd
import os
import pickle
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC

from collections import OrderedDict

from pdb import set_trace as st

import tensorflow as tf
from dovebirdia.filtering.kalman_filter import KalmanFilter

###########
# Functions
###########

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
   
    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
           ylim=[1.5,-0.5],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
            ax.text(j,i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] < thresh else "white")
    fig.tight_layout()

    return ax, cm

################################################################################
# PROCESS COMMAND LINE FLAGS
################################################################################

# define command line parser
parser = argparse.ArgumentParser()

# -cfg CONFIG_FILE
parser.add_argument("-c", "--config", dest = "config", help="Configuration File")
parser.add_argument("-r", "--results", dest = "results", help="Results q File Description")

# parse options from commandline and copy to dictionary
flag_dict = parser.parse_args().__dict__

# display config_diction
for k,v in flag_dict.items():

    print('{}: {}'.format(k,v))

# read flag_dict values
config_dir = flag_dict['config']

# read all config files
config_files = os.listdir(config_dir)

config_dicts = dict()

for config_file in config_files:

    config_name = os.path.splitext(config_file)[0].split('_')[-1]

    with open(config_dir + config_file, 'rb') as handle:

        config_dicts[config_name] = dill.load(handle)

################################
# Parameters
################################

# save related
SAVE_FIGURES = config_dicts['gen']['save_figures']

results_dir = config_dicts['gen']['results_dir']
sensor_response_dir = results_dir + 'sensor_response/'
max_response_dir = results_dir + 'max_response/' 
pca_dir = results_dir + 'pca/' 
cm_dir = results_dir + 'cm/'

try:
    
    os.makedirs(sensor_response_dir +'/figures/')
    os.makedirs(sensor_response_dir +'/data/')
    os.makedirs(pca_dir +'/figures/')
    os.makedirs(pca_dir +'/data/')
    os.makedirs(max_response_dir +'/data/')
    os.makedirs(cm_dir +'/figures/')
    os.makedirs(cm_dir +'/data/')
    
except:
    
    pass

# sensor response plots
NUM_COLORS = config_dicts['gen']['num_colors']

# Maximum samples to use since some experiments did not run the full 5000 samples
max_samples = config_dicts['gen']['max_samples']

# sampling frequency in Hertz
sample_freq = config_dicts['kf']['sample_freq']

# l2 norm condition for outlier.  At present if l2 norm is greater l2_norm_scale * mean_l2_norm classify as outlier
l2_norm_scale = config_dicts['gen']['l2_norm_scale']

# classification
sklearn_random_state = config_dicts['gen']['sklearn_random_state']

# resistance dataset key
ml_dataset = config_dicts['gen']['ml_dataset']

# location of parsed pickle files
pickle_dir = config_dicts['gen']['pickle_dir']

# number of pca components for machine learning
ml_pca_n_components = config_dicts['gen']['ml_pca_n_components']

# x0 and x1 for machine learning
x0 = config_dicts['gen']['x0']
x1 = config_dicts['gen']['x1']

##############
# Load Dataset
##############

# read names of pickle files
pickle_files = [ file for file in os.listdir(pickle_dir) if file.endswith('.pkl') ]

# list to hold trials read from pickle files
mixture_data = list()

for pickle_file in pickle_files:

    pickle_file_path = pickle_dir + '/' + pickle_file
    
    with open(pickle_file_path, 'rb') as handle:

        mixture_data.append(pickle.load(handle))

#################
# Remove Controls
#################

trials_to_del = list()

# negative controls
for trial_idx, trial in enumerate(mixture_data):

    concentration = trial['concentration']
    concentration_sum = np.asarray( [float(concen) for concen in concentration.split(',')] ).sum()

    if concentration_sum == 0.0:

        trials_to_del.append(trial_idx)
        
# positive controls
for trial_idx, trial in enumerate(mixture_data):

    name = list(trial['name'].split(','))
    concentration = list(trial['concentration'].split(','))
    concentration_sum = np.asarray( [float(concen) for concen in concentration] ).sum()

    ethanol_concentration = float(concentration[name.index('Ethanol')])

    if ethanol_concentration == 20.0 and concentration_sum == 20.0:

        trials_to_del.append(trial_idx)
        
# delete controls
for index in sorted(trials_to_del, reverse=True):

    del mixture_data[index]
        
print(len(mixture_data))

#####################
# Preprocessing Steps
#####################

# Delta R / R Standardization
for trial_idx, trial in enumerate(mixture_data):
    
    resistance = trial['resistance']
        
    baseline_res_mean = resistance[:500,:].mean(axis=0)
    delta_R_over_R = (resistance - baseline_res_mean) / baseline_res_mean
    
    mixture_data[trial_idx]['resistance_d'] = delta_R_over_R

# z-scoring
scaler = StandardScaler(with_mean=True, with_std=False)

for trial in mixture_data:
    
    scaler.fit(trial['resistance'][:trial['event_indices'],:] )  

    trial['resistance_z'] = scaler.transform(trial['resistance'])

if ml_dataset == 'resistance_kf0' or ml_dataset == 'resistance_kf1':
    
    # Kalman Filter
    kf_params = config_dicts['kf']

    for trial_idx, trial in enumerate(mixture_data):

        kf = KalmanFilter(kf_params)

        z = trial['resistance_z']

        kf_results = kf.fit(z)

        with tf.Session() as sess:

            res = sess.run(kf_results)

        trial['resistance_kf0'] = np.squeeze(res['x_hat_post'][:,0::2,:])
        trial['resistance_kf1'] = np.squeeze(res['x_hat_post'][:,1::2,:])

        tf.reset_default_graph()
    
#################################
# Select Trials Based on Analytes
#################################

df = pd.DataFrame(mixture_data)
hexane_idx = df.name.unique()[0].split(',').index('Hexane')
dmmp_idx = df.name.unique()[0].split(',').index('DMMP')

# lists to hold selected data
dmmp_data = list()
hexane_data = list()
dmmp_hexane_data = list()

for trial_idx, trial in enumerate(mixture_data):
    
    # trial concentration as np array
    concen = np.asarray(list(map(float, trial['concentration'].split(','))))
    
    # dmmp
    if concen[dmmp_idx] == concen.sum():

        mixture_data[trial_idx]['analyte_label'] = 0
        dmmp_data.append(trial)
        
    # hexane
    elif concen[hexane_idx] == concen.sum():
    
        mixture_data[trial_idx]['analyte_label'] = 1
        hexane_data.append(trial)
        
    # dmmp and hexane mixture
    elif concen[dmmp_idx]!=0 and concen[hexane_idx]!=0:
        
        mixture_data[trial_idx]['analyte_label'] = 2
        dmmp_hexane_data.append(trial)

#################
# Remove Outliers
#################

# # loop through dataframe, grouping by label
# for unique_label in df.label.unique()[:]:
    
#     # compute l2 norm for each member of group
#     group_l2_norm_list = list()

#     group_mean = np.dstack([ row[:max_samples,:] for row in df[df.label==unique_label].resistance.to_numpy()]).mean(axis=-1)
#     group_mean_z = np.dstack([ row[:max_samples,:] for row in df[df.label==unique_label].resistance_d.to_numpy()]).mean(axis=-1)
    
#     # loop over rows matching current label
#     for idx, row in df[df.label==unique_label].iterrows():

#         global_row_idx = df.index[df['csv_file'] == row['csv_file']]
#         group_l2_norm_list.append(np.linalg.norm(row['resistance_d'], axis=0))
#         df.at[global_row_idx[0], 'l2_norm'] = group_l2_norm_list[-1]

#     # mean l2 norm for current label
#     label_mean_l2_norm = np.asarray(group_l2_norm_list).mean(axis=0)
#     label_std_l2_norm = np.asarray(group_l2_norm_list).std(axis=0)
    
#     # second pass through group to compare each l2 norm with mean l2 norm
#     outlier_list = list()
    
#     for idx, row in df[df.label==unique_label].iterrows():
        
#         # for each sensor in label group, indices of sensors which meet outlier criteria
#         outliers_idx = np.where(np.asarray(row['l2_norm']>l2_norm_scale*label_mean_l2_norm)==True)

#         # replace outlier resistance values and mark as outlier
#         if len(outliers_idx[0]) > 0:

#             # row index of trial with outlier(s)
#             global_row_idx = df.index[df['csv_file'] == row['csv_file']]
            
# #             print(row['csv_file'].split('_')[-1], outliers_idx[0][0], unique_label)
            
#             replacement_mean = np.dstack([ res.resistance[:max_samples,:] 
#                                               for idx, res in df[df.label==unique_label].iterrows()
#                                                   if res.csv_file is not row['csv_file'] ]).mean(axis=-1)

#             replacement_mean_z = np.dstack([ res.resistance_d[:max_samples,:] 
#                                               for idx, res in df[df.label==unique_label].iterrows()
#                                                   if res.csv_file is not row['csv_file'] ]).mean(axis=-1)
            
#             # replace resistance_z with label group mean
#             df.at[global_row_idx[0], 'resistance_d'][:,outliers_idx[0]] = \
#                 replacement_mean_z[:max_samples,outliers_idx[0]]

#             # replace resistance with label group mean
#             df.at[global_row_idx[0], 'resistance'][:,outliers_idx[0]] = \
#                 replacement_mean[:max_samples,outliers_idx[0]]
            
#             # mark as outlier
#             df.at[global_row_idx[0], 'outlier'][outliers_idx[0]] = True

################################################################
# Average Responses for Each Analyte at Each Concentration Label
################################################################

dmmp_res_dict = dict()
hexane_res_dict = dict()
dmmp_hexane_res_dict = dict()

# key - list containing specific analyte data, value - dictionary to store average resistance values
analyte_global_dict = {
    'DMMP':(dmmp_idx,dmmp_data,dmmp_res_dict),
    'Hexane':(hexane_idx,hexane_data,hexane_res_dict),
    'DMMP+Hexane':(None,dmmp_hexane_data,dmmp_hexane_res_dict),
}

for analyte, idx_list_dict in analyte_global_dict.items():
    
    idx, ldata, ddata = idx_list_dict
    
    # DMMP
    df = pd.DataFrame(ldata)
    concentration = df.concentration.unique()

    for concen in concentration:

        concen_arr = np.asarray(list(map(float, concen.split(','))))
        concen_res = df[df.concentration==concen]['resistance_d']
            
        if idx == None:
            
            concen_arr_idx = str(int(concen_arr[dmmp_idx])) + '%, ' + str(int(concen_arr[hexane_idx])) + '%'
            ddata[concen_arr_idx] = concen_res.values.mean()
        
        else:
            
            concen_arr_idx = concen_arr[idx]
            ddata[concen_arr_idx] = concen_res[:max_samples].values.mean()

###################################
# Sensor Response and Max. Response 
###################################

# x_axis = np.linspace(0,max_samples//sample_freq,max_samples)

# for analyte, idx_list_dict in analyte_global_dict.items():

#     # max response csv file
#     max_response_csv_data = list()

#     # save to disk
#     csv_sensor_response_dict = dict()
#     csv_max_response_dict = dict()
    
#     idx, _, ddata = idx_list_dict

#     # loop over each sensor in mixture data
#     for sensor_idx, sensor in enumerate(mixture_data[0]['sensors'][:]):

#         max_response_csv_header = list()
        
#         max_response_csv_data_sensor = list()

# #         plt.figure(figsize=(6,6))

#         colors = iter(cm.Spectral(np.linspace(0,1,NUM_COLORS)))

#         for concen_idx, concen in enumerate(sorted(ddata.keys(), reverse=True)):

#             concentration = concen
#             resistance = ddata[concen][:max_samples,sensor_idx]
            
#             # max resistance data
#             max_response = resistance.max()
#             max_response_idx = np.argmax(resistance)
#             max_response_mean = resistance[max_response_idx-2:max_response_idx+3].mean()
#             csv_max_response_dict[concen] = max_response_mean
            
#             # sensor response data
#             csv_sensor_response_dict[concen] = resistance
            
#             # plot
#             if analyte == 'DMMP' or analyte == "Hexane":
                
#                 label = str(int(concentration)) + '%'
                
#             else:
                
#                 label = concentration
            
# #             plt.plot(x_axis, resistance, label=label, color=next(colors))

# #         plt.title('{analyte}, {sensor}'.format(analyte=analyte,sensor=sensor))
# #         plt.xlabel('Time (s)')
# #         plt.ylabel(r'  $\frac{\Delta R}{R}$', fontsize=16, labelpad=20).set_rotation(0)
# #         plt.grid()
# #         plt.legend()

# #         if SAVE_FIGURES:
        
# #             sensor_response_fig_file = 'sensor_response_{analyte}_{sensor}'.format(analyte=analyte.replace('+','_'),sensor=sensor.replace(' ','_'))
# #             plt.savefig(sensor_response_dir + 'figures/' + sensor_response_fig_file,
# #                         bbox_inches='tight',
# #                         dpi=300)

# #         else:
            
# #             plt.show()
        
# #         plt.close()
        
#         # sensor response csv file
#         sensor_response_csv_header = list()
#         sensor_response_csv_data = list()
        
#         for k,v in csv_sensor_response_dict.items():
            
#             sensor_response_csv_header.append(k)
#             sensor_response_csv_data.append(v)
            
#         sensor_response_csv_data = np.asarray(sensor_response_csv_data).T
        
#         sensor_response_csv_file = 'sensor_response_{analyte}_{sensor}.csv'.format(analyte=analyte.replace('+','_'),sensor=sensor.replace(' ','_'))
#         pd.DataFrame(sensor_response_csv_data).to_csv(sensor_response_dir + 'data/' + sensor_response_csv_file, 
#                                                       header=sensor_response_csv_header)

        
#         for k,v in csv_max_response_dict.items():
            
#             max_response_csv_header.append(k)
#             max_response_csv_data_sensor.append(v)
                       
#         max_response_csv_data_sensor.insert(0,sensor)
#         max_response_csv_data.append(max_response_csv_data_sensor)
            
#     # append sensor information to header and data
#     max_response_csv_header.insert(0,'Sensor')
        
#     max_response_csv_data = np.asarray(max_response_csv_data)

#     max_response_csv_file = 'max_response_{analyte}.csv'.format(analyte=analyte.replace('+','_'))

#     pd.DataFrame(max_response_csv_data).to_csv(max_response_dir + 'data/' + max_response_csv_file, 
#                                                header=max_response_csv_header)

#####
# PCA
#####

merged_analyte_data = np.hstack([dmmp_data , hexane_data, dmmp_hexane_data])
pca_resistance = np.asarray([trial[ml_dataset] for trial in merged_analyte_data]).reshape(len(merged_analyte_data),-1)
pca_resistance_y = np.asarray([ trial['analyte_label'] for trial in merged_analyte_data ])
pca_resistance_y_concen = np.asarray([trial['concentration'] for trial in merged_analyte_data])
pca_analyte_labels = [('DMMP',0), ('Hexane',1), ('DMMP+Hexane',2)]
pca_concentration_indices = mixture_data[0]['name']

df = pd.DataFrame(list(merged_analyte_data))
unique_mixture_concentrations = sorted([ ( float(concen.split(',')[2]),float(concen.split(',')[3]) ) for concen in list(df[df.analyte_label==2].concentration.unique()) ])
mixture_concentrations_dict = { k:v for v,k in enumerate(unique_mixture_concentrations)}

# custom color dictionary for DMMP+Hexane
non_mixture_colors = ['C0', 'C1', 'C2']
    
mixture_colors = cm.hsv(np.linspace(0,1,len(mixture_concentrations_dict)))

pca = PCA(n_components=3)
pca_resistance_t = pca.fit_transform(pca_resistance)

#plot
plt.figure(figsize=(6,6))

for idx, (pca_pt, pca_y) in enumerate(zip(pca_resistance_t,pca_resistance_y)):
      
    label = pca_analyte_labels[pca_y][0]
    
    if pca_y < 2:
        
        color = non_mixture_colors[pca_y]
        bbox_to_anchor=None
        
    else:
        
        concen = tuple(float(s) for s in pca_resistance_y_concen[idx].split(",")[2:4])
        color_idx = mixture_concentrations_dict[concen]
        color = mixture_colors[color_idx]
        label = label + ' ' + str(concen)
        bbox_to_anchor=(1.55,1.01)
        
    plt.scatter(pca_pt[0],
                pca_pt[1],
                label=label,
                color=color)
    
plt.title('Principal Component Analysis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),loc='upper right', bbox_to_anchor=bbox_to_anchor)

if SAVE_FIGURES:
        
    pca_fig_file = 'pca_plot'
    plt.savefig(pca_dir + 'figures/' + pca_fig_file,
                bbox_inches='tight',
                dpi=300)

else:
            
    plt.show()
            
plt.close()

# data
pca_csv_file = 'pca_data.csv'
pca_merged_data = np.hstack((
                             np.expand_dims(pca_resistance_y,axis=1),
                             np.expand_dims(pca_resistance_y_concen,axis=1),
                             pca_resistance_t))
pca_merged_data_header = ['Label', 'Concentration', 'PC1', 'PC2', 'PC3']
pd.DataFrame(pca_merged_data).to_csv(pca_dir + 'data/' + pca_csv_file, 
                                     header=pca_merged_data_header)

# write analyte to label mapping to file
pca_analyte_labels_file = 'pca_analyte_labels.txt'

with open(pca_dir + 'data/' + pca_analyte_labels_file, 'w') as file:
    
    for label in pca_analyte_labels:
    
        file.write('{label}\n'.format(label=label))
        
    file.write('\nAnalytes Corresponding to Concentration Indices in PCA data File:\n{pca_concentration_indices}'.               format(pca_concentration_indices=pca_concentration_indices))

###########################
# Multilabel Classification
###########################

# generate multilabel classification labels
pca_resistance_multi_y_list = list()

for y in pca_resistance_y:
    
    if y==2:
        
        label = [0,1]
    
    else:
        
        label = [y]
    
    pca_resistance_multi_y_list.append(label)
    
pca_resistance_multi_y = MultiLabelBinarizer().fit_transform(pca_resistance_multi_y_list)
print(pca_resistance_multi_y.shape)

# classification resistance
seconds = (x1-600)/sample_freq
config_dicts['gen']['exposure_time'] = seconds
clf_resistance = np.asarray([trial[ml_dataset][x0:x1,:] for trial in merged_analyte_data]).reshape(len(merged_analyte_data),-1)

# train/test split indicies
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=sklearn_random_state)
for tr_idx, ts_idx in sss.split(clf_resistance, pca_resistance_multi_y):

    train_idx = tr_idx
    test_idx = ts_idx

X_train, X_test, y_train, y_test = clf_resistance[train_idx], clf_resistance[test_idx], pca_resistance_multi_y[train_idx], pca_resistance_multi_y[test_idx]

# used to determine which concentrations were classified as such
merged_analyte_data_test = merged_analyte_data[test_idx]
    
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# pca
pca = PCA(n_components=ml_pca_n_components)
X_train_t = pca.fit_transform(X_train)
X_test_t = pca.transform(X_test)

class_names = np.asarray(['No DMMP', 'DMMP'])

clf_dict = {
    'knn':KNeighborsClassifier(n_neighbors=3),
    'rf':RandomForestClassifier(n_estimators=700, max_depth=5, random_state=sklearn_random_state),
}

clf_results_dict = dict()

for clf_name, clf in clf_dict.items():

    y_pred = clf.fit(X_train_t, y_train).predict(X_test_t)
    
    # DMMP specific true and pred
    y_dmmp_test = y_test[:,0]
    y_dmmp_pred = y_pred[:,0]

    tp = np.logical_and(y_dmmp_pred==y_dmmp_test, y_dmmp_test==1)
    tn = np.logical_and(y_dmmp_pred==y_dmmp_test, y_dmmp_test==0)
    fn = np.logical_and(y_dmmp_pred!=y_dmmp_test, y_dmmp_test==1)    
    fp = np.logical_and(y_dmmp_pred!=y_dmmp_test, y_dmmp_test==0)

    merged_analyte_data_test_tp_concen = [ trial['concentration'].split(',') for trial in merged_analyte_data_test[tp] ]
    merged_analyte_data_test_fp_concen = [ trial['concentration'].split(',') for trial in merged_analyte_data_test[fp] ]
    merged_analyte_data_test_tn_concen = [ trial['concentration'].split(',') for trial in merged_analyte_data_test[tn] ]
    merged_analyte_data_test_fn_concen = [ trial['concentration'].split(',') for trial in merged_analyte_data_test[fn] ]                                       
    cm_excel_header = merged_analyte_data_test[tp][0]['name'].split(',')
    
    with pd.ExcelWriter(cm_dir + 'data/true_positive_labels_{clf_name}.xlsx'.format(clf_name=clf_name)) as writer:

        try: pd.DataFrame(merged_analyte_data_test_tp_concen).to_excel(writer, sheet_name='True Positive', header=cm_excel_header)
        except: pass

        try: pd.DataFrame(merged_analyte_data_test_fp_concen).to_excel(writer, sheet_name='False Positive', header=cm_excel_header)
        except: pass
        
        try: pd.DataFrame(merged_analyte_data_test_tn_concen).to_excel(writer, sheet_name='True Negative', header=cm_excel_header)
        except: pass
        
        try: pd.DataFrame(merged_analyte_data_test_fn_concen).to_excel(writer, sheet_name='False Negative', header=cm_excel_header)
        except: pass

    accuracy = 1.0 - np.abs(y_dmmp_pred-y_dmmp_test).sum() / y_dmmp_test.shape[0]
    
    _, cm = plot_confusion_matrix(y_dmmp_test, y_dmmp_pred, 
                          classes=class_names,
                          title='Confusion Matrix for {clf}\n{seconds} sec.'
                                 ', {n_components} PCs'
                                 ', Acc. {accuracy:0.3f}'\
                                .format(clf=clf_name,
                                        seconds=seconds,
                                        n_components=ml_pca_n_components,
                                        accuracy=accuracy),
                          cmap=plt.cm.Blues)
    
    # save kf data
    clf_results_dict[clf_name + '_acc'] = accuracy
    clf_results_dict[clf_name + '_tn'] = cm.ravel()[0]
    clf_results_dict[clf_name + '_fp'] = cm.ravel()[1]
    clf_results_dict[clf_name + '_fn'] = cm.ravel()[2]
    clf_results_dict[clf_name + '_tp'] = cm.ravel()[3]
    
    if SAVE_FIGURES:

        cm_fig_file = 'cm_plot_{clf_name}'.format(clf_name=clf_name) 
        plt.savefig(cm_dir + 'figures/' + cm_fig_file,
                    bbox_inches='tight',
                    dpi=300)

    else:

        plt.show()

################################################################################
# CSV
################################################################################

# merge dictionaries in config_dicts and training_results_dict
merged_config_dicts = dict()

for config_dict in config_dicts.values():

    merged_config_dicts.update(config_dict)

# training results
merged_config_dicts.update(clf_results_dict)

# model id
merged_config_dicts.update({'model_id':os.getcwd().split('/')[-1].split('_')[-1]})

# change dictionary value to name if exists
for k,v in merged_config_dicts.items():

    try:

        merged_config_dicts[k] = v.__name__

    except:

        pass
        
results_file = os.getcwd() + '/results/clf_results.csv'

try:
    
    with open(results_file, 'a') as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=sorted(merged_config_dicts.keys()))

        if os.stat(results_file).st_size == 0:

            writer.writeheader()

        writer.writerow(merged_config_dicts)

except IOError:

    print("I/O error") 
