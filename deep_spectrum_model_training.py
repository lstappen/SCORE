import glob
import os
from random import seed, shuffle, randint
import numpy as np
import argparse
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from config import get_basic_config, get_basic_evaluation
from main import export_results

from statistics import mean
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, f1_score
from os.path import splitext, basename
from utils import deep_spectrum_file_name, create_folders


# For feature extraction
import pandas as pd
import librosa
import librosa.display

#For model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score


class Data:
    def __init__(self, splits, holdout):
        self.splits = splits
        self.split_idx = -1

        # list of numpy arrays, one element = one fold
        self.Xs_train = []
        self.Xs_val = []
        self.ys_train = []
        self.ys_val = []

        if holdout:
            splits -= 1
            self.X_train_holdout = None
            self.X_test_holdout = None
            self.y_train_holdout = None
            self.y_test_holdout = None

    # This could be used as an automatic iter
    def __iter__(self):
        return self

    def __next__(self):
        self.split += 1
        if self.split_idx < self.splits:
            return self.Xs_train[self.split_idx], self.Xs_val[self.split_idx], self.ys_train[self.split_idx], self.ys_val[self.split_idx]
        raise StopIteration



def kfold_holdout(X, y, groups, splits=5):
    group_kfold = GroupKFold(n_splits=splits)
    group_kfold.get_n_splits(X, y, groups)

    d_obj = Data(splits=splits, holdout=False)

    for train_index, test_index in group_kfold.split(X, y, groups):
        # inplace shuffeling
        shuffle(train_index)
        shuffle(test_index)
        d_obj.Xs_train.append(X[train_index])
        d_obj.Xs_val.append(X[test_index])
        d_obj.ys_train.append(y[train_index])
        d_obj.ys_val.append(y[test_index])

    return d_obj

#Interface
parser = argparse.ArgumentParser(description='Train an SVM model on deep spectrum features.')
#parser.add_argument('-f','--feature_type', type=str, dest='feature_type', action='store', default='compare',
#                    help='specify the type of features you want to generate')
parser.add_argument('-l','--label_type', type=str, dest='label_type', action='store', default='point',
                    help='specify the type of label you want to generate')
parser.add_argument('-g','--gender', type=str, dest='gender', action='store', default='mw',
                    help='gender of data (for score prediction)')
parser.add_argument('-t', '--splits', type=int, dest='SPLITS', action='store', default=5,
                    help='specify no of data splits')
args = parser.parse_args()


config = get_basic_config(feature_type = "ds", label_type = args.label_type, SPLITS = 5, HOLDOUT = False)
#single feature per file
data_path = deep_spectrum_file_name(config)

gender = args.gender

if args.label_type == "gender":
    gender = "mw"

# metric_name = "f1"
metric_name = "uar"

with open(data_path) as f:
    lines = f.readlines()
lines.pop(0)

group_no = 0
current_video_in_group = 0
max_videos_per_group = 2
current_video_name = ""
groups_list = []
X_list = []
Y_list = []
current_filename = ""
frame_features = []
frame_labels = []

for line in lines:
    if gender == "m" and "women" in line:
        continue
    if gender == "w" and not "women" in line:
        continue
    if gender == "mw":
        pass
    line_split = line.split(",")
    filename = line_split[0]
    filename_split = filename.split("/")
    if len(filename) <= 1:
        continue
    video_name = filename_split[1]
    groups_list.append(video_name.split("_")[0])
    #line_split.pop(0)
    line_split.pop(0)
    #line_split.pop(-1)
    X_list.append(line_split)
    if args.label_type == "point":
        if "no point" in filename:
            Y_list.append(0)
        else:
            Y_list.append(1)
    elif args.label_type == "gender":
        if "women" in line:
            Y_list.append(0)
        else:
            Y_list.append(1)

groups = np.array(groups_list, dtype = int)
X = np.array(X_list, dtype=float)
Y = np.array(Y_list, dtype=int)

print(X.shape)
print(Y.shape)


men_data_obj = kfold_holdout(X=X
                             , y=Y
                             , groups=groups
                             , splits=args.SPLITS)

C_values = [1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1, 1, 10]
lines = []
for C in C_values:
    print("C: {}".format(C))
    metric = []
    line=[str(C)]
    for fold in range(args.SPLITS):
        X_train = men_data_obj.Xs_train[fold]
        X_val = men_data_obj.Xs_val[fold]
        y_train = men_data_obj.ys_train[fold]
        y_val = men_data_obj.ys_val[fold]

        clf = svm.LinearSVC(C=C, random_state=0, max_iter=5000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        if metric_name == "f1":
            metric.append(f1_score(y_val, y_pred, average="micro"))
            print("Fold {}: F1 score: {}".format(fold, metric[fold]))
        elif metric_name == "uar":
            metric.append(recall_score(y_val, y_pred, average="macro")*100)
            print("Fold {}: UAR score: {}".format(fold, metric[fold]))

        line.append(str(metric[fold]))
    metric_array = np.array(metric)
    metric_mean = np.mean(metric_array)
    metric_Std = np.std(metric_array)
    if metric_name == "f1":
        print("Average F1 score: {}".format(metric_mean))
        print("Std F1 score: {}".format(metric_Std))
        print("--------------------------------------------------------------")
        lines.append(";".join(line) + "\n")
    elif metric_name == "uar":
        print("Average UAR score: {}".format(metric_mean))
        print("Std UAR score: {}".format(metric_Std))
        print("--------------------------------------------------------------")
        lines.append(";".join(line) + "\n")

config_results = get_basic_evaluation("svm", "test", "ds")
create_folders(config_results)
result_file_name = config_results['RESULT_PATH'] + "result" + "_ds_" + gender + "_" + args.label_type + ".csv"
with open(result_file_name, "w") as f:
    f.writelines(lines)