import os
import numpy as np
import pickle
import shutil
from keras.utils import to_categorical
import pandas as pd
### data preparation

# extract labels from filename
def labels_from_filename(files, label_type):
    labels = []
    for f in files:
        if label_type == 'point':
            if f.split('_')[-1].split('.')[-2] == 'no point':
                # no score = 0
                labels.append(0)
            else:
                # score = 1
                labels.append(1)
        elif label_type =='gender':
            if f.split('_')[-3] == 'f':
                # female = 0
                labels.append(0)
            else:
                # male = 1
                labels.append(1)           
    return np.array(labels)

# identify groups from filename
def groups_from_filename(files):
    groups = []
    for f in files:
        groups.append(int(f.split('_')[0]))
    return np.array(groups)

# helper function to parse file_name
def compare_lld_file_name(config, file_name):
    output_file_compare = config['FEATURE_PATH_COMPARE'] + file_name.split(os.sep)[-1] +  '.ComParE' + '.csv'
    output_file_lld  = lld_file_name(config, file_name)
    return output_file_compare, output_file_lld

def lld_file_name(config, file_name):
    output_file_lld  = config['FEATURE_PATH_LLD']  +  file_name.split(os.sep)[-1]  +  '.ComParE-LLD' + '.csv' 
    return output_file_lld

def spectro_file_name(config, file_name):
    output_file  = config['FEATURE_PATH_Spectro']  +  file_name.split(os.sep)[-1]  +  '.Spectro' + '.png' 
    return output_file

def deep_spectrum_file_name(config):
    output_file_ds = config['FEATURE_PATH_DS'] +  'ds' + '.csv'
    return output_file_ds


# helper function to parse file_name
def boaw_file_name(config, file_name):
    output_file_boaw = config['FEATURE_PATH_BoAW'] +  file_name.split(os.sep)[-1] + '.BoAW-' + str(config['csize']) + '.csv'
    return output_file_boaw

def egemaps_file_name(config, file_name):
    output_file_lld  = lld_file_name(config, file_name)
    output_file = config['FEATURE_PATH_eGemaps'] +  file_name.split(os.sep)[-1] + '.Egemaps' + '.csv'
    return output_file, output_file_lld

def create_folders_basic(config):
    for path in [v for k,v in config.items() if 'PATH' in k]:
        if not os.path.exists(path):
            os.makedirs(path)

def create_folders(config):
    if os.path.exists(config['EXPERIMENT_PATH']):
        shutil.rmtree(config['EXPERIMENT_PATH'])

    for path in [v for k,v in config.items() if 'PATH' in k]:
        if not os.path.exists(path):
            os.makedirs(path)

    #list_of_dictionaries = dataframe.to_dict('records')
    pd.DataFrame.from_dict(config, orient="index").to_csv(config['EXPERIMENT_PATH']+'config.csv')

def parameter_path(path, parameter):

    path_para = os.path.join(path, parameter_str(parameter)[:-1])
    if not os.path.exists(path_para):
        os.makedirs(path_para)
    return path_para

# merge to lists
def merge_nparr(list1, list2):

    return [np.concatenate((list1[i], list2[i]), 0) for i in range(len(list1))]

# dump pkl data objects
def dump_data_objects(config, all_data_obj, men_data_obj, women_data_obj):
    # ### Export all data objects
    with open(config['FEATURE_PATH_PKLS'] + 'women' + '.pkl', 'wb') as file:
        pickle.dump(women_data_obj, file)
    with open(config['FEATURE_PATH_PKLS'] + 'men' + '.pkl', 'wb') as file:
        pickle.dump(men_data_obj, file)
    with open(config['FEATURE_PATH_PKLS'] + 'all' + '.pkl', 'wb') as file:
        pickle.dump(all_data_obj, file)

# dump pkl data objects
def dump_data_object(config, all_data_obj):
    # ### Export all data objects
    with open(config['FEATURE_PATH_PKLS'] + 'gender' + '.pkl', 'wb') as file:
        pickle.dump(all_data_obj, file)

# load gender-specific pkl data object
def load_data_object(config, g):

    # Example: Load all data objects
    gender_mapping = {'all':'all','w':'women','m':'men','gender_pred_only':'gender'}

    print(g)
    print(gender_mapping[g])
    with open(config['FEATURE_PATH_PKLS'] + gender_mapping[g] + '.pkl', 'rb') as file:
        return pickle.load(file)

# here hot-encoding are missing in the pipeline!
def to_hot(y):
    return to_categorical(y, num_classes=num_labels)

# this depends on the model type
def X_reshape(X, num_channels):
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_channels)
    print('new shape: ', X.shape)
    return X, (X.shape[1], X.shape[2], X.shape[3])

def parameter_str(parameter):
    text = ''

    for k, v in parameter.items():
        text += str(k) + '_' + str(v) + '.'

    return text
