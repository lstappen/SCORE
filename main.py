from sklearn.preprocessing import MinMaxScaler
import argparse
import glob
import os
from random import seed
from datetime import datetime
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras.utils import to_categorical

# own modules
from config import get_cnn_config, get_svm_config, get_crnn_config, get_lstm_config
from utils import load_data_object, create_folders, to_hot, X_reshape, parameter_str
from measures import f1_all_folds, recall_all_folds, export_results, export_ys
import model_training, network_utils
from data import Data

## global
SEED = 23
seed(SEED)

## interface
parser = argparse.ArgumentParser(description='Prepare feature .pkl and run experiments')
parser.add_argument('-f','--feature_type', type=str, dest='feature_type', action='store', default='compare'
                    help='specify the type of features you want to use')
parser.add_argument('-l','--label_type', type=str, dest='label_type', action='store', default='point',
                    help='specify the type of label you want to use')
parser.add_argument('-m', '--model_type', type=str, dest='model_type', action='store', default='svm', 
                    help='name of model type')
parser.add_argument('-n','--experiment_name', type=str, dest='experiment_name', action='store', default='test', 
                    help='name of experiment')
parser.add_argument('-g','--gender', dest='gender', nargs='+', default=['all','m','w'], 
                    help='gender of data: all m w')
parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                    help='prints more output information if true')
args = parser.parse_args()

if __name__ == "__main__":

    if args.label_type == 'point':
        gender = args.gender
    else:
        gender = ['gender_pred_only']

    for g in gender:
        if args.model_type == 'lstm':
            config = get_lstm_config(model_type = args.model_type
                                    , experiment_name = args.experiment_name+'_'+g
                                    , feature_type = args.feature_type)
        elif 'crnn' in args.model_type:
            config = get_crnn_config(model_type = args.model_type
                                    , experiment_name = args.experiment_name+'_'+g
                                    , feature_type = args.feature_type)
        elif args.model_type == 'svm':
            config = get_svm_config(model_type = args.model_type
                                    , experiment_name = args.experiment_name+'_'+g
                                    , feature_type = args.feature_type)
        # Some other architectures we experimented with but did not make it into the paper
        elif args.model_type == 'cnn':
            config = get_cnn_config(model_type = args.model_type
                                    , experiment_name = args.experiment_name+'_'+g
                                    , feature_type = args.feature_type)
        elif args.model_type == 'cnn_end':
            config = get_crnn_config(model_type = args.model_type
                                    , experiment_name = args.experiment_name+'_'+g
                                    , feature_type = args.feature_type)
        else:
            print("No config for model {} found".format(args.model_type))
            exit()

        create_folders(config)

        data_obj = load_data_object(config, g)

        y_pred_folds = {}
        y_devel_folds = {}

        for parameter in config['parameter_list']:
            parameter_text = parameter_str(parameter)
            print('[run] ', parameter_text)
            y_pred_folds[parameter_text] = []
            y_devel_folds[parameter_text] = []

            start = datetime.now()
            # train and evaluate one model for each fold - 4 splits for training / one hold out
            for fold_no in range(len(data_obj.Xs_train)):
                print(len(data_obj.Xs_train))
                X_train, X_devel = data_obj.Xs_train[fold_no], data_obj.Xs_val[fold_no]

                ######
                # ### select model and data "after care" depending on model
                if config['model_type'] == 'lstm':
                    if args.verbose:
                        print("model specific data preparation")

                    print(X_train.shape)
                    input_shape = (X_train.shape[1], X_train.shape[2])

                    y_train, y_devel = to_categorical(data_obj.ys_train[fold_no], num_classes=config['num_labels']), to_categorical(data_obj.ys_val[fold_no], num_classes=config['num_labels'])

                    model = network_utils.create_lstm_model(config, parameter['learning_rate'],input_shape,config['num_labels'])    

                    y_devel_folds[parameter_text], y_pred_folds[parameter_text] = model_training.train_network(config, fold_no
                                                                                            , model, parameter
                                                                                            , X_train , y_train
                                                                                            , X_devel, y_devel
                                                                                            , y_devel_folds[parameter_text], y_pred_folds[parameter_text])

                elif config['model_type'] == 'cnn':

                    if args.verbose:
                        print("model specific data preparation")
                    y_train, y_devel = to_categorical(data_obj.ys_train[fold_no], num_classes=config['num_labels']), to_categorical(data_obj.ys_val[fold_no], num_classes=config['num_labels'])
                    
                    X_train, input_shape = X_reshape(X_train, config['num_channels'])
                    X_devel, _ = X_reshape(X_devel, config['num_channels'])

                    if args.verbose:
                        print("X_train: ", X_train.shape, ", X_devel: ", X_devel.shape)
                        print("Y_train: ", y_train.shape, ", Y_devel ", y_devel.shape)

                    model = network_utils.create_cnn_model(parameter['learning_rate'],input_shape,config['num_labels'])    

                    y_devel_folds[parameter_text], y_pred_folds[parameter_text] = model_training.train_network(config, fold_no
                                                                                            , model, parameter
                                                                                            , X_train , y_train
                                                                                            , X_devel, y_devel
                                                                                            , y_devel_folds[parameter_text], y_pred_folds[parameter_text])
                elif config['model_type'] == 'cnn_end':
                    if args.verbose:
                        print("model specific data preparation")
                    y_train, y_devel = to_categorical(data_obj.ys_train[fold_no], num_classes=config['num_labels']), to_categorical(data_obj.ys_val[fold_no], num_classes=config['num_labels'])
                    if args.verbose:
                        print("X_train: ", X_train.shape, ", X_devel: ", X_devel.shape)
                        print("Y_train: ", y_train.shape, ", Y_devel ", y_devel.shape)   
                    
                    X_train, input_shape = X_reshape(X_train, config['num_channels'])
                    X_devel, _ = X_reshape(X_devel, config['num_channels'])

                    model = network_utils.create_cnn_end2end_model(config, parameter['learning_rate'],input_shape,config['num_labels']) 
                    y_devel_folds[parameter_text], y_pred_folds[parameter_text] = model_training.train_network(config, fold_no
                                                                                            , model, parameter
                                                                                            , X_train , y_train
                                                                                            , X_devel, y_devel
                                                                                            , y_devel_folds[parameter_text], y_pred_folds[parameter_text])                    
                    

                elif 'crnn' in config['model_type']:

                    if args.verbose:
                        print("model specific data preparation")
                    y_train, y_devel = to_categorical(data_obj.ys_train[fold_no], num_classes=config['num_labels']), to_categorical(data_obj.ys_val[fold_no], num_classes=config['num_labels'])
                   

                    if args.verbose:
                        print("X_train: ", X_train.shape, ", X_devel: ", X_devel.shape)
                        print("Y_train: ", y_train.shape, ", Y_devel ", y_devel.shape)

                    if config['model_type'] == 'crnn':
                        X_train, input_shape = X_reshape(X_train, config['num_channels'])
                        X_devel, _ = X_reshape(X_devel, config['num_channels'])

                        model = network_utils.create_crnn_small_model(parameter['learning_rate'],input_shape,config['num_labels'])   
                    elif  config['model_type'] == 'crnn_end':
                        #X_train = X_train_ #.reshape(X_train_.shape)
                        #X_devel = X_devel_#.reshape(1, X_devel_.shape)
                        input_shape = (X_train.shape[1], X_train.shape[2])
                        print(input_shape)

                        model = network_utils.create_crnn_end2end_model(config, parameter['learning_rate'],input_shape,config['num_labels']) 
                    y_devel_folds[parameter_text], y_pred_folds[parameter_text] = model_training.train_network(config, fold_no
                                                                                            , model, parameter
                                                                                            , X_train , y_train
                                                                                            , X_devel, y_devel
                                                                                            , y_devel_folds[parameter_text], y_pred_folds[parameter_text])
                    del model

                elif config['model_type'] == 'svm':

                    y_train, y_devel = data_obj.ys_train[fold_no], data_obj.ys_val[fold_no]
                    
                    if X_train.ndim > 2 and X_train.shape[1] > 1: #features, we have to get rid off the time dimension
                        if config['svm_seq_agg'] == 'mean':
                            X_train = np.mean(X_train, axis=1)
                            X_devel = np.mean(X_devel, axis=1)
                        elif config['svm_seq_agg'] == 'middle':
                            X_train = X_train[:,int(X_train.shape[1]/2),:]
                            X_devel = X_devel[:,int(X_devel.shape[1]/2),:]
                        elif config['svm_seq_agg'] == 'flatten':
                            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
                            X_devel = X_devel.reshape(X_devel.shape[0], X_devel.shape[1] * X_devel.shape[2])
                    print('SVM X_train input format: ', X_train.shape)
                    new_X_train = X_train.reshape(X_train.shape[0], X_train.shape[-1])
                    new_X_devel = X_devel.reshape(X_devel.shape[0], X_devel.shape[-1])

                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(new_X_train)
                    X_devel = scaler.transform(new_X_devel)

                    if args.verbose:
                        print("X_train: ", X_train.shape, ", X_devel: ", X_devel.shape)
                        print("Y_train: ", y_train.shape, ", Y_devel ", y_devel.shape)

                    y_devel_folds[parameter_text], y_pred_folds[parameter_text] = model_training.train_svm(config, fold_no, parameter 
                                                                                            , X_train, y_train
                                                                                            , X_devel, y_devel
                                                                                            , y_devel_folds[parameter_text], y_pred_folds[parameter_text])

                else:
                    print("Model {} not defined".format(config['model_type']))
                    exit()


            # average results of all fold experiments
            f1_all_folds(y_devel_folds[parameter_text], y_pred_folds[parameter_text])

            # average results of all fold experiments
            recall_all_folds(y_devel_folds[parameter_text], y_pred_folds[parameter_text])

            export_ys(config, parameter, y_devel_folds[parameter_text], y_pred_folds[parameter_text])

        export_results(config, config['parameter_list'], y_devel_folds, y_pred_folds)


