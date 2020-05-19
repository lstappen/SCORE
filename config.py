import os

# data processing
def get_basic_config(feature_type, label_type, SPLITS, HOLDOUT):

    config = {}

    config['feature_type'] = feature_type #input_type to feature_type
    config['label_type'] = label_type
    config['SPLITS'] = SPLITS
    config['HOLDOUT'] = HOLDOUT

    config['DATA_PATH'] = '.' + os.sep + 'data'
    config['FEATURE_PATH'] = '.' + os.sep + 'features'
    config['FEATURE_PATH_COMPARE'] = config['FEATURE_PATH'] + os.sep + 'compare/'
    config['FEATURE_PATH_LLD'] = config['FEATURE_PATH'] + os.sep + 'lld' + os.sep
    config['FEATURE_PATH_Spectro'] = config['FEATURE_PATH'] + os.sep + 'spectrogram' + os.sep
    config['FEATURE_PATH_BoAW'] = config['FEATURE_PATH'] + os.sep + 'boaw' + os.sep
    config['FEATURE_PATH_eGemaps'] = config['FEATURE_PATH'] + os.sep + 'egemaps' + os.sep
    config['FEATURE_PATH_PKLS'] = config['FEATURE_PATH'] + os.sep + 'pkls' + os.sep + config['feature_type'] + os.sep
    config['GENDER'] = ['women', 'men']

    # Modify openSMILE paths here:
    config['SMILEexe'] = '~/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
    config['SMILEconf'] = '~/opensmile-2.3.0/config/ComParE_2016.conf'
    config['egemapsconf'] = '~/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf'
    config['openXBOW'] = './tools/openXBOW.jar'

    if config['feature_type'] == 'mfcc':
        config = get_mfcc_config(config)

    if config['feature_type'] == 'lld':
        config = get_lld_config(config)

    if 'boaw' in config['feature_type']:
        # modify BoAW settings here
        config = get_lld_config(config)
        config = get_bow_config(config)

    if config['feature_type'] == 'raw':
        config = get_raw_audio_config(config)
    
    # modify DeepSpecturm settings here

    return config

def get_mfcc_config(config):
    # modify MFCC settings here
    config['num_components'] = 40  # number of MFCCs to return
    config['num_timesteps_mfcc'] = 44  # frames
    config['mfcc_max_pad_len'] = config['num_timesteps_mfcc']
    config['num_channels'] = 1

    return config

def get_lld_config(config):
    # modify LDDs settings here
    config['num_timesteps_lld'] = 100  # frames
    config['lld_max_pad_len'] = config['num_timesteps_lld']

    return config

def get_lld_config(config):
    # modify LDDs settings here
    config['num_timesteps_lld'] = 224  # frames
    config['lld_max_pad_len'] = config['num_timesteps_lld']

    return config

def get_bow_config(config):
    # modify LDDs settings here
    config['csize'] = int(config['feature_type'].split('_')[-1])  # frames
    
    if config['csize'] not in [125, 250, 500, 1000, 2000]:
        print("csize not in [125, 250, 500, 1000, 2000] ")
        exit()
    #config['lld_max_pad_len'] = config['num_timesteps_lld']

    return config

def get_raw_audio_config(config):

    config['frame_rate'] = 40 # 40 ms 1 sec
    config['audio_length'] = 1000
    config['sample_rate'] = 16000
     
    return config


# experiments
# overall config
def get_basic_evaluation(model_type, experiment_name, feature_type):
    config = {}

    config['feature_type'] = feature_type #input_type to feature_type
    config['model_type'] = model_type 
    config['experiment_name'] = experiment_name #input_type to feature_type
    config['FEATURE_PATH'] = '.' + os.sep + 'features'
    config['FEATURE_PATH_PKLS'] = config['FEATURE_PATH'] + os.sep + 'pkls' + os.sep + config['feature_type'] + os.sep

    config['EXPERIMENT_PATH'] = '.' + os.sep + 'experiments' + os.sep + experiment_name + os.sep
    config['PLOT_PATH'] = config['EXPERIMENT_PATH'] + 'plots' + os.sep
    config['MODEL_PATH'] = config['EXPERIMENT_PATH'] + 'model' + os.sep
    config['RESULT_PATH'] = config['EXPERIMENT_PATH'] + 'results' + os.sep

    config['num_labels'] = 2

    return config

def get_lstm_config(model_type, experiment_name, feature_type):

    config = get_basic_evaluation(model_type, experiment_name, feature_type)
    config = get_mfcc_config(config)

    if config['feature_type'] == 'mfcc':
        config['lstm1_n'] = 40
        config['lstm2_n'] = 40//2
        config['parameter_list'] = [{
            'num_epochs': 200, 
            'num_batch_size': 32,
            'learning_rate' : 0.0001
        },
        ]
    elif config['feature_type'] == 'lld':
        config['lstm1_n'] = 50
        config['lstm2_n'] = 30
        config['parameter_list'] = [{
            'num_epochs': 200, 
            'num_batch_size': 32,
            'learning_rate' : 0.0001
        },
        ]
    elif config['feature_type'] == 'raw':
        config['lstm1_n'] = 50
        config['lstm2_n'] = 30
        config['parameter_list'] = [{
            'num_epochs': 200, 
            'num_batch_size': 32,
            'learning_rate' : 0.0001
        },
        ]
    else:
        config['lstm1_n'] = 50
        config['lstm2_n'] = 30
        config['parameter_list'] = [{
            'num_epochs': 200, 
            'num_batch_size': 32,
            'learning_rate' : 0.0001
        },
        ]
    # experiment parameter          identified as best working parameters:



    return config


# cnn specific config
def get_cnn_config(model_type, experiment_name, feature_type):

    config = get_basic_evaluation(model_type, experiment_name, feature_type)
    config = get_mfcc_config(config)


    # experiment parameter          identified as best working parameters:

    config['parameter_list'] = [{
        'num_epochs': 100, 
        'num_batch_size': 10,
        'learning_rate' : 0.0001
    },
    {
        'num_epochs': 100, 
        'num_batch_size': 64,
        'learning_rate' : 0.00001
    }]


    return config

# cnn specific config
def get_crnn_config(model_type, experiment_name, feature_type):
    # parameter_list identified as best working parameters

    config = get_basic_evaluation(model_type, experiment_name, feature_type)
    config = get_mfcc_config(config)

    if config['feature_type'] == 'mfcc':
        config['Conv1_filters'] = 10
        config['Conv1_kernel_size'] = 6
        config['Conv2_filters'] = 20
        config['Conv2_kernel_size'] = 8
        config['Conv3_filters'] = 40
        config['Conv3_kernel_size'] = 10
        config['lstm1_n'] = 40
        config['lstm2_n'] = 40//2
        config['parameter_list'] = [
                    { #I
                    'num_epochs': 500, 
                    'num_batch_size': 16,
                    'learning_rate' : 0.00005
                    },
                    { #II
                    'num_epochs': 500, 
                    'num_batch_size': 16,
                    'learning_rate' : 0.0001
                    },
                    { #IV
                    'num_epochs': 500, 
                    'num_batch_size': 32,
                    'learning_rate' : 0.001
                    },  
                    ] 
        #get_parameter_optimisation()

    elif config['feature_type'] == 'lld':
        config['Conv1_filters'] = 30
        config['Conv1_kernel_size'] = 10
        config['Conv2_filters'] = 30
        config['Conv2_kernel_size'] = 8
        config['Conv3_filters'] = 40
        config['Conv3_kernel_size'] = 10
        config['lstm1_n'] = 50
        config['lstm2_n'] = 30
        # Models from the paper
        config['parameter_list'] = [
                    { #I
                    'num_epochs': 500, 
                    'num_batch_size': 16,
                    'learning_rate' : 0.00005
                    },
                    { # II
                    'num_epochs': 500, 
                    'num_batch_size': 16,
                    'learning_rate' : 0.0001
                    },
                    { # IV
                    'num_epochs': 500, 
                    'num_batch_size': 32,
                    'learning_rate' : 0.001
                    },  
                    ] 
        #get_parameter_optimisation()
 
    if config['feature_type'] == 'raw': #NOT USED
        config['Conv1_filters'] = 40
        config['Conv1_kernel_size'] = 2
        config['Conv2_filters'] = 50
        config['Conv2_kernel_size'] = 4
        config['Conv3_filters'] = 0
        config['Conv3_kernel_size'] = 8
        config['lstm1_n'] = 50
        config['lstm2_n'] = 40//2
        config['parameter_list'] = get_parameter_optimisation()
        # [{
        # 'num_epochs': 500, 
        # 'num_batch_size': 50,
        # 'learning_rate' : 0.0001
        # },]
    else: # spectorgrams
        config['Conv1_filters'] = 10
        config['Conv1_kernel_size'] = 6
        config['Conv2_filters'] = 20
        config['Conv2_kernel_size'] = 8
        config['Conv3_filters'] = 40
        config['Conv3_kernel_size'] = 10
        config['lstm1_n'] = 40
        config['lstm2_n'] = 40//2
        config['parameter_list'] = [
                    { #III
                    'num_epochs': 500, 
                    'num_batch_size': 16,
                    'learning_rate' : 0.001
                    },
                    { # II
                    'num_epochs': 500, 
                    'num_batch_size': 16,
                    'learning_rate' : 0.0001
                    },
                    { # IV
                    'num_epochs': 500, 
                    'num_batch_size': 32,
                    'learning_rate' : 0.001
                    },  
                    ] 


    


    return config

def get_parameter_optimisation():
    # test [{'num_epochs': 1, 
    # 'num_batch_size': 16, 
    # 'learning_rate' : 0.01}]

    return  [{'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 16, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 32, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.01},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.001},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 0.0001},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 5e-05},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 1e-05},
    {'num_epochs': 500, 
     'num_batch_size': 64, 
     'learning_rate' : 1e-05},]


# svm specific config
def get_svm_config(model_type, experiment_name, feature_type):

    config = get_basic_evaluation(model_type, experiment_name, feature_type)

    config['svm_seq_agg'] = 'middle' # mean or middle
    # experiment parameter          identified as best working parameters:
    config['parameter_list'] = [
    {
        'C': 1e-6,
        'max_iter': 10000 
    },
    {
        'C': 1e-5,
        'max_iter': 10000 
    },
    {
        'C': 1e-4,
        'max_iter': 10000  
    },
    {
        'C': 1e-3,
        'max_iter': 10000  
    },
    {
        'C': 1e-2,
        'max_iter': 10000  
    },
    {
        'C': 1e-1,
        'max_iter': 10000  
    },
    {
        'C': 1e0,
        'max_iter': 10000  
    },
    {
        'C': 10,
        'max_iter': 10000 
    }
    ]

    return config

    