"""
Copyright (C) 2022 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
You can be released from the terms, and requirements of the Academic public license by purchasing a commercial license.
"""
from __future__ import absolute_import, division, print_function

from collections import defaultdict as dd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

import pickle
import os
import time
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, average_precision_score, precision_recall_curve, roc_auc_score, auc, confusion_matrix
from scipy.special import softmax
import scipy
import sys

import pandas as pd

# print(tf.__version__)

dataset_name_list = ['svhn-cnn', 'cifar10-resnet32']

# print('FIXME: only doing svhn-cnn')
# dataset_name_list = ['svhn-cnn']

TRIALS = 10

# also known as recall or true positive rate (TPR)
def sensitivity(tp, fn):
    return tp / (tp + fn)

# Also known as selectivity, or true negative rate (TNR)
def specificity(tn, fp):
    return tn / (tn + fp)

# beta > 1 gives more weight to specificity, while beta < 1 favors
# sensitivity. For example, beta = 2 makes specificity twice as important as
# sensitivity, while beta = 0.5 does the opposite.
def f_score_sens_spec(sens, spec, beta=1.0):

    # return (1 + beta**2) * ( (precision * recall) / ( (beta**2 * precision) + recall ) )

    return (1 + beta**2) * ( (sens * spec) / ( (beta**2 * sens) + spec ) )

def threshold_scores(preds, tau):
    if isinstance(preds, np.ndarray):
        return np.where(preds>tau, 1.0, 0.0)

    elif torch.is_tensor(preds):
        return torch.where(preds>tau, 1.0, 0.0)

    else:
        raise TypeError(f"ERROR: preds is expected to be of type (torch.tensor, numpy.ndarray) but is type {type(preds)}")
  
def determine_threshold(misclf_labels, scores, max_threshold_step=.01):
    
    # determine how many elements we need for a pre-determined spacing
    # between thresholds. taken from:
    # https://stackoverflow.com/a/70230433
    num = round((scores.max() - scores.min()) / max_threshold_step) + 1 
    thresholds = np.linspace(scores.min(), scores.max(), num, endpoint=True)

    # compute performance over thresholds
    threshold_to_metric = {}
    for tau in thresholds:

        predicted_labels = threshold_scores(scores, tau)

        tn, fp, fn, tp = confusion_matrix(misclf_labels, predicted_labels).ravel()
        
        specificity_value = specificity(tn, fp)
        sensitivity_value = sensitivity(tp, fn)

        f_beta_spec_sens = f_score_sens_spec(sensitivity_value,
                                             specificity_value, beta=1.0)

        # print(f'tau: {tau:.6f}, spec: {specificity_value:.4f}, sens: {sensitivity_value:.4f}, f_beta: {f_beta_spec_sens:.4f}, balance: {abs(specificity_value-sensitivity_value):.4f}')
        
        threshold_to_metric[tau] = f_beta_spec_sens

    # determine best threshold:
    best_item = max(threshold_to_metric.items(), key=lambda x: x[1])

    best_tau, best_metric = best_item
    # print(f'best | tau: {best_tau:.6f}, metric: {best_metric:.4f}', end='\n\n')

    return best_tau

# TODO: use top 3 max difference between mean_correct_valid and
# mean_incorrect_valid to pick 3 best performing models for each dataset

def AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name, NN_info="64+64"):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_run{}.pkl'.format(dataset_name, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        # if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
        #     continue
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        if metric_name == "AP-error" or metric_name == "AUPR-error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = -exp_result["mean_test"].reshape(-1)
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = exp_result["mean_test"].reshape(-1)
        if metric_name == "AP-error" or metric_name == "AP-success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "AUROC":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for {}: {}".format(framework_variant, AP_list))
    return AP_list

def AP_calculation_topN(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, top_num, metrics):
    AP_list = []

    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_run{}.pkl'.format(dataset_name, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        # if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
        #     continue

        # not used in our experiments
        if framework_variant == "GP_inputOnly":
            print('GP_inputOnly')
            
            # result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
            # with open(result_file_name, 'rb') as result_file:
            #     exp_result = pickle.load(result_file)
            # if metric_name == "AP-error" or metric_name == "AUPR-error":
            #     y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            #     y_score_test = -exp_result["mean_test"].reshape(-1)
            # else:
            #     y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            #     y_score_test = exp_result["mean_test"].reshape(-1)
            # if metric_name == "AP-error" or metric_name == "AP-success":
            #     AP_list.append(average_precision_score(y_true_test, y_score_test))
            # elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            #     precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            #     AP_list.append(auc(recall, precision))
            # elif metric_name == "AUROC":
            #     AP_list.append(roc_auc_score(y_true_test, y_score_test))
            #print("AP_list for {}: {}".format(framework_variant, AP_list))
        else:
            
            trial_num = TRIALS
            AP_list_tmp = []

            for trial in range(trial_num):
                
                add_info=''
                result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))

                with open(result_file_name, 'rb') as result_file:
                    exp_result_tmp = pickle.load(result_file)
                    # print('reading:', result_file_name)

                # setup scores / labels
                if metric_name == "ap_error" or metric_name == "aupr_error":
                    y_true_test = (exp_result_tmp["test_labels"]!=np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = -exp_result_tmp["mean_test"].reshape(-1)

                    y_true_valid = (exp_result_tmp["valid_labels"]!=np.argmax(exp_result_tmp["valid_NN_predictions"], axis=1))
                    y_score_valid = -exp_result_tmp["mean_valid"].reshape(-1)
                    
                else:
                    y_true_test = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = exp_result_tmp["mean_test"].reshape(-1)

                    y_true_valid = (exp_result_tmp["valid_labels"]==np.argmax(exp_result_tmp["valid_NN_predictions"], axis=1))
                    y_score_valid = exp_result_tmp["mean_valid"].reshape(-1)

                # compute metric
                if metric_name == "ap_error" or metric_name == "ap_success":
                    AP_list_tmp.append(average_precision_score(y_true_test, y_score_test))
                elif metric_name == "aupr_error" or metric_name == "aupr_success":
                    precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
                    AP_list_tmp.append(auc(recall, precision))
                elif metric_name == "roc_auc":
                    AP_list_tmp.append(roc_auc_score(y_true_test, y_score_test))

                elif metric_name == 'sensitivity':

                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    AP_list_tmp.append(sensitivity(tp, fn))
                    
                elif metric_name == 'specificity':

                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    AP_list_tmp.append(specificity(tn, fp))
                    
                elif metric_name == 'f_score_spec_sens@beta=1.0':

                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    spec = specificity(tn, fp)
                    sens = sensitivity(tp, fn)
                    
                    AP_list_tmp.append(f_score_sens_spec(sens, spec, beta=1.0))
                    
                elif metric_name == 'f_score_spec_sens@beta=2.0':
                    
                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    spec = specificity(tn, fp)
                    sens = sensitivity(tp, fn)
                    
                    AP_list_tmp.append(f_score_sens_spec(sens, spec, beta=2.0))
                    
                else:
                    raise Exception(f'ERROR: Unrecognized metric name: {metric_name}')
                    
                #  seperate opt:
                add_info = "+separate_opt"
                result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
                
                with open(result_file_name, 'rb') as result_file:
                    exp_result_tmp = pickle.load(result_file)
                    # print('reading:', result_file_name)

                # setup scores / labels:
                if metric_name == "ap_error" or metric_name == "aupr_error":
                    y_true_test = (exp_result_tmp["test_labels"]!=np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = -exp_result_tmp["mean_test"].reshape(-1)
                else:
                    y_true_test = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = exp_result_tmp["mean_test"].reshape(-1)
                    
                if metric_name == "ap_error" or metric_name == "ap_success":
                    AP_list_tmp.append(average_precision_score(y_true_test, y_score_test))
                elif metric_name == "aupr_error" or metric_name == "aupr_success":
                    precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
                    AP_list_tmp.append(auc(recall, precision))
                elif metric_name == "roc_auc":
                    AP_list_tmp.append(roc_auc_score(y_true_test, y_score_test))

                elif metric_name == 'sensitivity':

                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    AP_list_tmp.append(sensitivity(tp, fn))
                    
                elif metric_name == 'specificity':

                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    AP_list_tmp.append(specificity(tn, fp))
                    
                elif metric_name == 'f_score_spec_sens@beta=1.0':

                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    spec = specificity(tn, fp)
                    sens = sensitivity(tp, fn)
                    
                    AP_list_tmp.append(f_score_sens_spec(sens, spec, beta=1.0))
                    
                elif metric_name == 'f_score_spec_sens@beta=2.0':
                    
                    tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

                    predicted_test_labels = threshold_scores(y_score_test, tau)
                    
                    tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

                    spec = specificity(tn, fp)
                    sens = sensitivity(tp, fn)
                    
                    AP_list_tmp.append(f_score_sens_spec(sens, spec, beta=2.0))
                    
                else:
                    raise Exception(f'ERROR: Unrecognized metric name: {metric_name}')

            # print('list len:', len(AP_list_tmp))
            AP_list.append(np.mean(np.sort(AP_list_tmp, axis=None)[-top_num:]))

        # print("AP_list_top{} for {}: {}".format(top_num, framework_variant, AP_df[metric_name].values))
        #print("AP_list_top{} mean for {}: {}".format(top_num, framework_variant, np.mean(AP_list)))

    return AP_list[0]

def AP_class_max(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_run{}.pkl'.format(dataset_name, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        # if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
        #     continue
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        softmax_predictions = np.max(softmax(exp_result["test_NN_predictions"], axis=1), axis=1).reshape(-1)
        if metric_name == "AP-error" or metric_name == "AUPR-error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = -softmax_predictions
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = softmax_predictions
        if metric_name == "AP-error" or metric_name == "AP-success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "AUROC":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for class max: {}".format(AP_list))
    return AP_list

# def AP_class_difference(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name):
#     AP_list = []
#     for run in range(RUNS):
#         result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_run{}.pkl'.format(dataset_name, run))
#         with open(result_file_name, 'rb') as result_file:
#             exp_info = pickle.load(result_file)
#         # if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
#         #     continue
#         result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
#         with open(result_file_name, 'rb') as result_file:
#             exp_result = pickle.load(result_file)
#         test_check = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
#         softmax_predictions = softmax(exp_result["test_NN_predictions"], axis=1)
#         softmax_predictions_sorted = np.sort(softmax_predictions)
#         class_diff = softmax_predictions_sorted[:,-1] - softmax_predictions_sorted[:,-2]
#         if metric_name == "AP-error" or metric_name == "AUPR-error":
#             y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
#             y_score_test = -class_diff
#         else:
#             y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
#             y_score_test = class_diff
#         if metric_name == "AP-error" or metric_name == "AP-success":
#             AP_list.append(average_precision_score(y_true_test, y_score_test))
#         elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
#             precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
#             AP_list.append(auc(recall, precision))
#         elif metric_name == "AUROC":
#             AP_list.append(roc_auc_score(y_true_test, y_score_test))
#         #print("AP_list for class difference: {}".format(AP_list))
#     return AP_list


def AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name):
    AP_list = []
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_run{}.pkl'.format(dataset_name, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
                    
        trial = 0
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
        with open(result_file_name, 'rb') as result_file:
            exp_result = pickle.load(result_file)
        
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),SOTA_dir_name,('{}_exp_info_'+SOTA_algo_name+'_run{}.pkl').format(dataset_name, run))
        try:
            with open(result_file_name, 'rb') as result_file:
                SOTA_exp_info = pickle.load(result_file)
                # print('reading:', result_file_name)
        except:
            print("skip {} run{} for {}".format(dataset_name, run, SOTA_algo_name))
            continue

        # setup scores / labels
        if metric_name == "ap_error" or metric_name == "aupr_error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            
            y_true_valid = (exp_result["valid_labels"]!=np.argmax(exp_result["valid_NN_predictions"], axis=1))
            
            if SOTA_algo_name == "TrustScore":
                y_score_test = -(SOTA_exp_info["trust_score_test"].reshape(-1))
            else:
                y_score_test = -(SOTA_exp_info["moderator_test_NN_predictions"].reshape(-1))
                y_score_valid = -(SOTA_exp_info["moderator_valid_NN_predictions"].reshape(-1))
                
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_true_valid = (exp_result["valid_labels"]==np.argmax(exp_result["valid_NN_predictions"], axis=1))
            
            if SOTA_algo_name == "TrustScore":
                y_score_test = SOTA_exp_info["trust_score_test"].reshape(-1)
            else:
                y_score_test = SOTA_exp_info["moderator_test_NN_predictions"].reshape(-1)
                y_score_valid = SOTA_exp_info["moderator_valid_NN_predictions"].reshape(-1)


        # compute metric:
        if metric_name == "ap_error" or metric_name == "ap_success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "aupr_error" or metric_name == "aupr_success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "roc_auc":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for {}: {}".format(SOTA_algo_name, AP_list))

        elif metric_name == 'sensitivity':

            tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

            predicted_test_labels = threshold_scores(y_score_test, tau)
                    
            tn, fp, fn, tp = confusion_matrix(y_true_test,
                                                      predicted_test_labels).ravel()

            AP_list.append(sensitivity(tp, fn))
                    
        elif metric_name == 'specificity':

            tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

            predicted_test_labels = threshold_scores(y_score_test, tau)
                    
            tn, fp, fn, tp = confusion_matrix(y_true_test,
                                              predicted_test_labels).ravel()

            AP_list.append(specificity(tn, fp))
                    
        elif metric_name == 'f_score_spec_sens@beta=1.0':

            tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

            predicted_test_labels = threshold_scores(y_score_test, tau)
            
            tn, fp, fn, tp = confusion_matrix(y_true_test,
                                              predicted_test_labels).ravel()

            spec = specificity(tn, fp)
            sens = sensitivity(tp, fn)
                    
            AP_list.append(f_score_sens_spec(sens, spec, beta=1.0))
                    
        elif metric_name == 'f_score_spec_sens@beta=2.0':
                    
            tau = determine_threshold(y_true_valid, y_score_valid, max_threshold_step=.01)

            predicted_test_labels = threshold_scores(y_score_test, tau)
                    
            tn, fp, fn, tp = confusion_matrix(y_true_test,
                                              predicted_test_labels).ravel()

            spec = specificity(tn, fp)
            sens = sensitivity(tp, fn)
                    
            AP_list.append(f_score_sens_spec(sens, spec, beta=2.0))
 
        else:
            raise Exception(f'ERROR: Unrecognized metric name: {metric_name}')
        
    return AP_list[0]

for dataset_index in range(len(dataset_name_list)):

    dataset_name = dataset_name_list[dataset_index]

    RUNS = 1
    dir_name = f"Results/{dataset_name}"

    metric_name_list = ["ap_error", "ap_success", "aupr_error", "aupr_success", "roc_auc",
                        'sensitivity', 'specificity',
                        'f_score_spec_sens@beta=1.0', 'f_score_spec_sens@beta=2.0']
    
    # metric_name = metric_name_list[0]

    metrics_dict = dict()
    
    print("Showing Results for {}".format(dataset_name))
    print("Showing Results for dir: {}".format(dir_name))

    algo_spec = "moderator_residual_target"
    add_info = ""

    kernel_type = "RBF+RBF"
    framework_variant = "GP_corrected"
    top_num = 3

    for metric_name in metric_name_list:    
        metrics_dict[metric_name] = AP_calculation_topN(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, top_num, metric_name)

    json.dump(metrics_dict,
              open(os.path.join(dir_name, 'RED_metrics.json'), 'w'))

    print('RED metrics:')
    for k, v in metrics_dict.items():
        print(f'{k}: {v}')
    print(end='\n\n')

    # Introspection net
    SOTA_algo_name = "Introspection-Net"
    SOTA_dir_name = f"Results/{dataset_name}"
    SOTA_metrics_dict = dict()
    
    for metric_name in metric_name_list:    
        SOTA_metrics_dict[metric_name] = AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name)

    json.dump(SOTA_metrics_dict,
              open(os.path.join(dir_name, f'{SOTA_algo_name}_metrics.json'), 'w'))

    print(SOTA_algo_name, 'metrics')
    for k, v in SOTA_metrics_dict.items():
        print(f'{k}: {v}')
    print(end='\n\n')
