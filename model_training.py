# For model
import os
from keras.utils import to_categorical
from measures import f1
from sklearn import svm
import numpy as np
from utils import parameter_path
from network_utils import visualise_training

def train_network(config, fold_no, model, parameter, X_train, y_train, X_devel, y_devel, y_devel_folds, y_pred_folds):
    from keras.callbacks import ModelCheckpoint, EarlyStopping

    num_epochs = parameter['num_epochs']                          
    num_batch_size = parameter['num_batch_size']

    checkpoint_path = parameter_path(config['MODEL_PATH'], parameter)

    # export best model
    checkpointer = ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'weights.best.' + config['feature_type'] + '.' + config['experiment_name'] + '_'+  'fold.' + str(fold_no) +'.hdf5'),
                                   verbose=0, save_best_only=True)

    # stop training if validation loss does not improve for 20 rounds
    stopper = EarlyStopping(monitor='val_loss', patience=40
                            , verbose=0
                            , mode='min'
                            , restore_best_weights=True)

    history = model.fit(X_train, y_train
                        , batch_size=num_batch_size
                        , epochs=num_epochs
                        , validation_data=(X_devel, y_devel)
                        , callbacks=[checkpointer, stopper]
                        , verbose=0)

    #prepare data for F1-score measurement
    #y_pred = model.evaluate(X_devel, y_devel, verbose=0)
    y_pred = model.predict(X_devel)
    y_pred = np.around(y_pred, decimals=0)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    y_devel = np.array(y_devel)
    y_devel = np.where(y_devel > 0.5, 1, 0)
    print("Results for fold number ", fold_no, " is: ", f1(y_devel, y_pred))

    # Store trained model predictions
    y_pred_folds.append(y_pred)
    y_devel_folds.append(y_devel)

    # plot results
    visualise_training(config, history, fold_no, parameter)
    del history

    return y_devel_folds, y_pred_folds   


def train_svm(config, fold_no, parameter, X_train, y_train, X_devel, y_devel, y_devel_folds, y_pred_folds):

    comp = parameter['C']
    max_iter = parameter['max_iter']
    print('\nComplexity {0:.6f}'.format(comp))

    clf = svm.LinearSVC(C=comp, random_state=0, max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_devel)

    #print("Results for fold number ", fold_no, " is: ", f1(y_devel, y_pred))

    # Store trained model predictions
    y_pred_folds.append(y_pred)
    y_devel_folds.append(y_devel)

    return y_devel_folds, y_pred_folds    