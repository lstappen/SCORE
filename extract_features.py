import argparse
import glob
import os
from random import seed, shuffle, randint
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import imageio
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

## global
SEED = 23
seed(SEED)

# own modules
from config import get_basic_config
from utils import labels_from_filename, groups_from_filename, compare_lld_file_name, create_folders_basic, merge_nparr, dump_data_objects, dump_data_object, boaw_file_name, egemaps_file_name, spectro_file_name
from data import Data
from PIL import Image



## interface
parser = argparse.ArgumentParser(description='Prepare feature .pkl')
parser.add_argument('-f','--feature_type', type=str, dest='feature_type', action='store', default='compare', required=True,
                    help='specify the type of features you want to generate')
parser.add_argument('-l','--label_type', type=str, dest='label_type', action='store', default='point',
                    help='specify the type of label you want to generate')
parser.add_argument('-t', '--splits', type=int, dest='SPLITS', action='store', default=5, 
                    help='specify no of data splits')
parser.add_argument('--holdout', dest='HOLDOUT', action='store_true', default=False,
                    help='hold one partition back for test only')
args = parser.parse_args()



# feature extraction
## COMPaRE
def extract_features_compare(config, file_name, overwrite = False):

    output_file_compare, output_file_lld = compare_lld_file_name(config, file_name)
    
    file_name_console =  '"' + '.'+ os.sep + file_name  + '"'
    output_file_console = '"' + output_file_compare + '"'
    output_file_lld_console = '"' + output_file_lld + '"'

    # Extract openSMILE features for the file (standard ComParE and LLD-only)
    cmd = config['SMILEexe'] + ' -C ' + config['SMILEconf']  + ' -I ' + file_name_console + ' -instname ' + file_name_console + ' -csvoutput '+ output_file_console + ' -timestampcsv 1 -lldcsvoutput ' + output_file_lld_console + ' -appendcsvlld 1'
    
    if os.path.exists(output_file_compare) and os.path.exists(output_file_lld) and not overwrite:
        run = False
    else:
        run = True
        if os.path.exists(output_file_compare):
            print('remove: ' + output_file_compare)
            os.remove(output_file_compare)
        if os.path.exists(output_file_lld):
            print('remove: ' + output_file_lld)
            os.remove(output_file_lld)

    if run:
        # execute
        return_os = os.system(cmd)
    else:
        return_os = 0

    if return_os > 0:
        print('Failure executing: ' + cmd)
    else:
        compare_df = pd.read_csv(output_file_compare, sep = ';')          
        compare = compare_df.as_matrix(columns=compare_df.columns[2:])
        print("[X]  " + file_name)        
        return compare

def extract_features_llds(config, file_name, overwrite = False):
    output_file_compare, output_file_lld = compare_lld_file_name(config, file_name)
    
    file_name_console =  '"' + '.'+ os.sep + file_name  + '"'
    output_file_console = '"' + output_file_compare + '"'
    output_file_lld_console = '"' + output_file_lld + '"'

    # Extract openSMILE features for the file (standard ComParE and LLD-only)
    cmd = config['SMILEexe'] + ' -C ' + config['SMILEconf']  + ' -I ' + file_name_console + ' -instname ' + file_name_console + ' -csvoutput '+ output_file_console + ' -timestampcsv 1 -lldcsvoutput ' + output_file_lld_console + ' -appendcsvlld 1'
    
    if os.path.exists(output_file_lld) and not overwrite:
        run = False
    else:
        run = True
        if os.path.exists(output_file_lld):
            print('remove: ' + output_file_lld)
            os.remove(output_file_lld)

    if run:
        # execute
        return_os = os.system(cmd)
    else:
        return_os = 0

    if return_os > 0:
        print('Failure executing: ' + cmd)
    else:
        lld_df = pd.read_csv(output_file_lld, sep = ';')          
        lld = lld_df.as_matrix(columns=lld_df.columns[2:])
        print("[X]  " + file_name)  
        lld_padded = np.zeros((config['num_timesteps_lld'], lld.shape[1]))
        lld_padded[:lld.shape[0],:] =lld[:min(config['num_timesteps_lld'],lld.shape[0]),:]

        return lld_padded

def extract_features_egemaps(config, file_name, overwrite = False):

    output_file_egemaps, output_file_lld = egemaps_file_name(config, file_name)
    
    file_name_console =  '"' + '.'+ os.sep + file_name  + '"'
    output_file_console = '"' + output_file_egemaps + '"'
    output_file_lld_console = '"' + output_file_lld + '"'


    # Extract eGemaps features for the file (standard ComParE and LLD-only)
    config_options = config['SMILEexe'] + ' -configfile ' + config['egemapsconf']  + ' -inputfile ' + file_name_console + ' -instname ' + file_name_console + ' -csvoutput '+ output_file_console
    final_options = ' -appendcsvlld 0 -timestampcsvlld 1 -timestampcsv 1'  #-headercsvlld 1 -timestampcsv 1 
    cmd = config_options + final_options
    #cmd2 = '-lldcsvoutput ' + output_file_lld_console + ' -appendcsvlld 1'   
    #opensmile_call = config['SMILEexe'] + ' ' + opensmile_options + ' -inputfile ' + file_name_console + ' ' +  + ' -instname ' + file_name_console + ' -csvoutput '+ output_file_console  # (disabling htk output)
    if os.path.exists(output_file_egemaps):# and os.path.exists(output_file_lld) and not overwrite:
        run = False
    else:
        run = True
        if os.path.exists(output_file_egemaps):
            print('remove: ' + output_file_egemaps)
            os.remove(output_file_egemaps)
        # if os.path.exists(output_file_lld):
        #     print('remove: ' + output_file_lld)
        #     os.remove(output_file_lld)

    if run: # execute
        return_os = os.system(cmd)
    else:
        return_os = 0

    if return_os > 0:
        print('Failure executing: ' + cmd)
    else:
        egemaps_df = pd.read_csv(output_file_egemaps, sep = ';')  
        egemaps = egemaps_df.as_matrix(columns=egemaps_df.columns[2:])
        #print(egemaps)
                
        #exit()
        #print("[X]  " + file_name)    
        #print(egemaps.shape)    
        return egemaps

# BoAW
def extract_features_BoAW(config, file_name, overwrite = False):

    # make sure LLDs are created
    extract_features_compare(config, file_name, overwrite = False)
    
    _, output_file_lld = compare_lld_file_name(config, file_name)
    output_file_boaw = boaw_file_name(config, file_name)

    output_file_lld_console = '"' + output_file_lld + '"'
    output_file_boaw_console = '"' + output_file_boaw + '"'

    # Compute BoAW representations from openSMILE LLDs
    num_assignments = 10

    xbow_config = '-i ' + output_file_lld_console + ' -attributes nt1[65]2[65] -o ' + output_file_boaw_console
    #if partition=='train':
    xbow_config += ' -standardizeInput -size ' + str(config['csize']) + ' -a ' + str(num_assignments) + ' -log -B codebook_' + str(config['csize'])
    #else:
    #xbow_config += ' -b codebook_' + str(config['csize'])
    cmd = 'java -Xmx20000m -jar ' + config['openXBOW'] +' -writeName ' + xbow_config

    if os.path.exists(output_file_boaw) and not overwrite:
        run = False
    else:
        run = True
        if os.path.exists(output_file_boaw):
            print('remove: ' + output_file_boaw)
            os.remove(output_file_boaw)

    if run:
        return_os = os.system(cmd)
    else:
        return_os = 0

    if return_os > 0:
        print('Failure executing: ' + cmd)
    else:
        boaw_df = pd.read_csv(output_file_boaw, sep = ';') 

        boaw = boaw_df.as_matrix(columns=boaw_df.columns[2:])
        print("[X]  " + file_name)     
    return boaw

def extract_features_mfcc(config, file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=config['num_components'])  # shape=(n_mfcc, t)
    pad_width = config['mfcc_max_pad_len'] - mfccs.shape[1]
    mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    mfccs_transposed = np.transpose(mfccs_padded)  # flip time and feature axis!

    if int(file_name.split(os.sep)[-1].split('_')[0]) == 1:  # only first file
        print("parse no ", file_name.split(os.sep)[-1].split('_')[0], ': ', file_name)
        print('sampling rate: ', sample_rate)
        print('full: ', mfccs.shape)
        print('padded: ', mfccs_padded.shape)
        print('transposed: ', mfccs_transposed.shape)

    return mfccs_transposed

def load_features_spectrogram(config, file_name):

    file_name_new = spectro_file_name(config, file_name)

    im = (np.array(Image.open(file_name_new).convert('L')).astype(np.float32) - 128) / 128 

    return im

def extract_raw_signal(config, file_name):

    fix_feature_per_window = 640

    audio_clip = AudioFileClip(str(file_name))
    clip = audio_clip.set_fps(config['sample_rate'])
    num_samples = int(clip.fps * config['frame_rate'])

    data_frame = np.array([i for i in clip.iter_frames()])
    data_frame = np.squeeze(data_frame)
    data_frame = data_frame.mean(1) # two channels
    frames = data_frame.astype(np.float32)
    
    shape = data_frame.shape

    chunk_size = int(config['audio_length']/config['frame_rate']) # split audio file to chuncks of 40 ms
    audio_padded = np.pad(data_frame, (0, chunk_size - data_frame.shape[0] % chunk_size), 'constant')
    audio_padded = np.reshape(audio_padded, (-1, chunk_size)).astype(np.float32)

    audio_transposed = np.transpose(audio_padded)  # flip time and feature axis!

    padded = np.zeros((chunk_size, fix_feature_per_window))
    padded[:,:min(fix_feature_per_window,audio_transposed.shape[1])] = audio_transposed[:,:min(fix_feature_per_window,audio_transposed.shape[1])]

    return padded


def extract_gender_specific_features(config, files, labels, groups):
    # features 
    features_male = []
    features_female = []

    for g in config['GENDER']:
        # iterate through all files and extract features to a new list
        for fi, label in zip(files[g], labels[g]):
            # print("fi: ", str(fi)," ,la: ", str(la))
            if config['feature_type'] == 'compare':
                features = extract_features_compare(config, fi)  
            elif config['feature_type'] == 'lld':
                features = extract_features_llds(config, fi)   
            elif config['feature_type'] == 'egemaps':
                features = extract_features_egemaps(config, fi)  
            elif 'boaw' in config['feature_type']:
                features = extract_features_BoAW(config, fi)  
            elif config['feature_type'] == 'mfcc':
                features = extract_features_mfcc(config, fi) 
            elif config['feature_type'] == 'spectro':
                features = load_features_spectrogram(config, fi)     
            elif config['feature_type'] == 'raw': 
                features = extract_raw_signal(config, fi)       
            else:
                print('feature_type ', config['feature_type'], ' not valid.')
                exit()

            if (g == 'women'):
                features_female.append([features, label])
            else:
                features_male.append([features, label])
    return features_male, features_female


## Fold partitioning with holdout
def kfold_holdout(X, y, groups, splits, holdout):
    group_kfold = GroupKFold(n_splits=splits)
    group_kfold.get_n_splits(X, y, groups)

    d_obj = Data(splits=splits, holdout=holdout)

    for train_index, test_index in group_kfold.split(X, y, groups):
        # inplace shuffeling
        shuffle(train_index)
        shuffle(test_index)
        # generate folds
        if holdout == True:
            if d_obj.X_test_holdout is None:
                # first folds are for test only
                d_obj.X_train_holdout, d_obj.X_test_holdout = X[train_index], X[test_index]
                d_obj.y_train_holdout, d_obj.y_test_holdout = y[train_index], y[test_index]
                store_test_index = test_index
            else:
                # holdout idx if re-occuring in train
                train_index = [x for x in train_index if x not in store_test_index]
                d_obj.Xs_train.append(X[train_index])
                d_obj.Xs_val.append(X[test_index])
                d_obj.ys_train.append(y[train_index])
                d_obj.ys_val.append(y[test_index])

        elif holdout == False:
            d_obj.Xs_train.append(X[train_index])
            d_obj.Xs_val.append(X[test_index])
            d_obj.ys_train.append(y[train_index])
            d_obj.ys_val.append(y[test_index])
        else:
            print("Something is wrong here")
            exit()

    return d_obj


def data_objects_fusion(obj1, obj2):
    # Initialize data with parameters form top
    obj = Data(splits=config['SPLITS'], holdout=config['HOLDOUT'])
    # simply merge all
    obj.Xs_train = merge_nparr(obj1.Xs_train, obj2.Xs_train)
    obj.Xs_val = merge_nparr(obj1.Xs_val, obj2.Xs_val)
    obj.ys_train = merge_nparr(obj1.ys_train, obj2.ys_train)
    obj.ys_val = merge_nparr(obj1.ys_val, obj2.ys_val)

    if config['HOLDOUT']:
        obj.X_train_holdout = np.concatenate((obj1.X_train_holdout, obj2.X_train_holdout), axis=None)
        obj.X_test_holdout = np.concatenate((obj1.X_test_holdout, obj2.X_test_holdout), axis=None)
        obj.y_train_holdout = np.concatenate((obj1.y_train_holdout, obj2.y_train_holdout), axis=None)
        obj.y_test_holdout = np.concatenate((obj1.y_test_holdout, obj2.y_test_holdout), axis=None)
    return obj

def extract_file_data(config):
    # create new lists for input files
    files, files_prep, labels, groups = {}, {}, {}, {}

    # prepare input data
    for g in config['GENDER']:
        path = os.path.join('data', g, '**', '*.wav')
        # complete path: 'data\\women\\9_Q_nDx06rr58_874\\9_Q_nDx06rr58_874_f_9_point.wav'
        files[g] = [f for f in glob.glob(path, recursive=True)]
        # complete file names: '15_xNGJWMA4Tpg_2573_f_10_point.wav'
        files_prep[g] = [f.split(os.path.sep)[-1] for f in files[g]]
        # label: point/ no point in [0,1]: [1 1 1 1 1 1 0 0 0 ...]
        labels[g] = labels_from_filename(files_prep[g],config['label_type'])
        # group: file as a number 1, 15, 22, ...: [15 15 15 15 15 15 ...]
        groups[g] = groups_from_filename(files_prep[g])
    return files, labels, groups


def create_data_objects(config, groups, features_male, features_female):

    # Convert into a Panda dataframe
    featuresdf_male = pd.DataFrame(features_male, columns=['feature', 'class_label'])
    featuresdf_female = pd.DataFrame(features_female, columns=['feature', 'class_label'])
    print('Finished feature extraction from ', len(featuresdf_male), ' files')
    print('Finished feature extraction from ', len(featuresdf_female), ' files')

    # Convert features and corresponding classification labels into numpy arrays
    male_np_array_X = np.array(featuresdf_male.feature.tolist())
    male_np_array_y = np.array(featuresdf_male.class_label.tolist())

    female_np_array_X = np.array(featuresdf_female.feature.values.tolist())
    female_np_array_y = np.array(featuresdf_female.class_label.tolist())

    # Partitioning for women and men
    g = 'women'
    women_data_obj = kfold_holdout(
                                    X=female_np_array_X
                                   , y=female_np_array_y
                                   , groups=groups[g]
                                   , splits=config['SPLITS']
                                   , holdout=config['HOLDOUT'] )

    g = 'men'
    men_data_obj = kfold_holdout( 
                                  X=male_np_array_X
                                 , y=male_np_array_y
                                 , groups=groups[g]
                                 , splits=config['SPLITS']
                                 , holdout=config['HOLDOUT'])


    #merge genders
    all_data_obj = data_objects_fusion(women_data_obj, men_data_obj)

    return all_data_obj, men_data_obj, women_data_obj



if __name__ == "__main__":

    config = get_basic_config(feature_type = args.feature_type, label_type = args.label_type, SPLITS = args.SPLITS, HOLDOUT = args.HOLDOUT)
    print(config)
    create_folders_basic(config)

    print('[Info] Extract file data ')
    files, labels, groups = extract_file_data(config)
    print('[Info] Extract features from file... ')
    features_male, features_female = extract_gender_specific_features(config, files, labels, groups)
    print('[Info] Build data objects including folds, features and labels')
    all_data_obj, men_data_obj, women_data_obj = create_data_objects(config
                                                                        , groups
                                                                        , features_male
                                                                        , features_female)
    print('[Info] Export objects')
    if config['label_type'] == 'gender':
        dump_data_object(config, all_data_obj)
    else:
        dump_data_objects(config, all_data_obj, men_data_obj, women_data_obj)



