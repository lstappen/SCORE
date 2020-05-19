from sklearn.metrics import f1_score, recall_score
import csv
import numpy as np
import pickle
import os

from utils import parameter_str, parameter_path

round_precisions = 2

def f1(y_true, y_pred):
    return round(100 * f1_score(y_true, y_pred, average='micro'),round_precisions)

def recall(y_true, y_pred):
    return round(100 * recall_score(y_true, y_pred, average='macro'),round_precisions)

def mean_std(fold_results):
    return round(np.mean(fold_results),round_precisions), round(np.std(fold_results),round_precisions)
    
# F1 accuracy measurement
def f1_all_folds(y_true_folds, y_pred_folds):
    fold_results = [f1(y_true_folds[i], y_pred_folds[i]) for i in range(len(y_true_folds))]
    #print(fold_results)
    mean, std = mean_std(fold_results)
    # print("Average F1 score of all folds : ", mean)
    return fold_results, mean, std

# F1 accuracy measurement
def recall_all_folds(y_true_folds, y_pred_folds):
    fold_results = [recall(y_true_folds[i], y_pred_folds[i]) for i in range(len(y_true_folds))]
    #print(fold_results)
    mean, std = mean_std(fold_results)
    # print("Average UAR score of all folds : ", mean)

    return fold_results, mean, std

def avgresult(results):

    #remove configname, avg, std
    avg = [v[-2] for v in results if v[0].startswith('C')]
    std = [v[-1] for v in results if v[0].startswith('C')]

    if(len(avg)>0):
        avg_avg, _ = mean_std(avg)
        avg_std, _ = mean_std(std)

        top_results = ['avg result'] + ['' for i in range(len(results[0][1:-2]))] + [avg_avg, avg_std]
        results.append(top_results)
    return results

def aggresult(results, op):

    folds = results[0][1:-2]
    config_result = [v[1:-2] for v in results if v[0].startswith('C')] #remove configname, avg, std

    if(len(config_result)>0): 
        top_result_folds = []
        for fold in range(len(folds)):
            fv = [config[fold] for config in config_result]
            if op == 'max':
                top_result_folds.append(max(fv))
            if op == 'avg':
                top_result_folds.append(sum(fv) / len(fv))

        mean, std = mean_std(top_result_folds)
        top_results = ['max result'] + top_result_folds + [mean, std]
        results.append(top_results)
    return results

def export_ys(config, parameter, y_devel_folds, y_pred_folds):
    
    path = parameter_path(config['RESULT_PATH'], parameter)

    with open(os.path.join(path, 'y_devel_folds' + '.pkl'), 'wb') as file:
        pickle.dump(y_devel_folds, file)
    with open(os.path.join(path, 'y_pred_folds' + '.pkl'), 'wb') as file:
        pickle.dump(y_pred_folds, file)

def export_results(config, parameter_list, y_devel_folds, y_pred_folds):

    splits = len(y_pred_folds[list(y_pred_folds.keys())[0]])
    # dynamischer header 
    header = ['P_text']
    for no in range(splits):
        header.append('fold ' + str(no + 1))
    header.append('avg')
    header.append('std')
    
    #TO-DO: Clean-up to one for f1 and UAR
    for measure in ['uar','f1']:
        results = []
        results.append(header)
        for parameter in config['parameter_list']:
            result_run = []
            parameter_text = parameter_str(parameter)
            result_run.append(parameter_text)
            if measure == 'f1':
                fold_results, mean, std = f1_all_folds(y_devel_folds[parameter_text], y_pred_folds[parameter_text])
            elif measure == 'uar':
                fold_results, mean, std = recall_all_folds(y_devel_folds[parameter_text], y_pred_folds[parameter_text])
            else:
                print("Measure not definied")
                exit()

            for fold in fold_results:
                result_run.append(fold)
            result_run.append(mean)
            result_run.append(std)
            results.append(result_run)
        
        results = avgresult(results)
        results = aggresult(results, op='max')

        results = [["{:.3f}".format(j) if not isinstance(j, str) else j for j in i] for i in results]
        with open(config['RESULT_PATH'] + 'results_'+measure+'.csv', 'w') as file:
            wr = csv.writer(file,quoting=csv.QUOTE_NONE) #, dialect='excel'
            wr.writerows(results)



