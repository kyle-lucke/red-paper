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
import numpy as np
from sklearn.metrics import mean_absolute_error, average_precision_score, precision_recall_curve, roc_auc_score, auc
from scipy.special import softmax
import scipy
import sys

import pandas as pd

# print(tf.__version__)

# dataset_name_list = ['svhn-cnn', 'cifar10-resnet32']
print('FIXME: only doing svhn-cnn')
dataset_name_list = ['svhn-cnn']

TRIALS = 3
print(f'FIXME: ONLY USING {TRIALS} TRIALS')

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

def AP_calculation_topN(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, top_num, metric_name):
    AP_list = []
    AP_dict = dd(list)
    
    for run in range(RUNS):
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_run{}.pkl'.format(dataset_name, run))
        with open(result_file_name, 'rb') as result_file:
            exp_info = pickle.load(result_file)
        # if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
        #     continue
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
            max_difference = -100
            AP_list_tmp = []
            AP_dict_tmp = dd(list)
            difference_list_test_tmp = []
            difference_list_train_tmp = []
            noise_variance_list_tmp = []
            signal_noise_ratio_list_tmp = []

            for trial in range(trial_num):
                
                add_info=''
                result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
                
                AP_dict_tmp['fname'].append(result_file_name)
                
                with open(result_file_name, 'rb') as result_file:
                    exp_result_tmp = pickle.load(result_file)
                test_check = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                if metric_name == "AP-error" or metric_name == "AUPR-error":
                    y_true_test = (exp_result_tmp["test_labels"]!=np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = -exp_result_tmp["mean_test"].reshape(-1)
                else:
                    y_true_test = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = exp_result_tmp["mean_test"].reshape(-1)
                if metric_name == "AP-error" or metric_name == "AP-success":
                    AP_dict_tmp[metric_name].append(average_precision_score(y_true_test, y_score_test))
                    AP_list_tmp.append(average_precision_score(y_true_test, y_score_test))
                elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
                    precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
                    AP_dict_tmp[metric_name].append(auc(recall, precision))
                    AP_list_tmp.append(auc(recall, precision))
                elif metric_name == "AUROC":
                    AP_dict_tmp[metric_name].append(roc_auc_score(y_true_test, y_score_test))
                    AP_list_tmp.append(roc_auc_score(y_true_test, y_score_test))
                
                difference_list_test_tmp.append(exp_result_tmp["mean_correct_test"] - exp_result_tmp["mean_incorrect_test"])
                difference_list_train_tmp.append(exp_result_tmp["mean_correct_train"] - exp_result_tmp["mean_incorrect_train"])

                AP_dict_tmp['mean_difference_valid'].append(exp_result_tmp["mean_correct_valid"] - exp_result_tmp["mean_incorrect_valid"])

                # AP_dict_tmp['mean_difference_train'].append(exp_result_tmp["mean_correct_train"] - exp_result_tmp["mean_incorrect_train"])
                
                noise_variance_list_tmp.append(exp_result_tmp["hyperparameter"][-1])
                signal_noise_ratio_list_tmp.append((exp_result_tmp["hyperparameter"][1]+exp_result_tmp["hyperparameter"][3])/exp_result_tmp["hyperparameter"][-1])

                add_info = "+separate_opt"
                result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))

                AP_dict_tmp['fname'].append(result_file_name)
                
                with open(result_file_name, 'rb') as result_file:
                    exp_result_tmp = pickle.load(result_file)

                test_check = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                
                if metric_name == "AP-error" or metric_name == "AUPR-error":
                    y_true_test = (exp_result_tmp["test_labels"]!=np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = -exp_result_tmp["mean_test"].reshape(-1)
                else:
                    y_true_test = (exp_result_tmp["test_labels"]==np.argmax(exp_result_tmp["test_NN_predictions"], axis=1))
                    y_score_test = exp_result_tmp["mean_test"].reshape(-1)
                    
                if metric_name == "AP-error" or metric_name == "AP-success":
                    AP_dict_tmp[metric_name].append(average_precision_score(y_true_test, y_score_test))
                    AP_list_tmp.append(average_precision_score(y_true_test, y_score_test))
                elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
                    precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
                    AP_dict_tmp[metric_name].append(auc(recall, precision))
                    AP_list_tmp.append(auc(recall, precision))
                elif metric_name == "AUROC":
                    AP_dict_tmp[metric_name].append(roc_auc_score(y_true_test, y_score_test))
                    AP_list_tmp.append(roc_auc_score(y_true_test, y_score_test))

                                
                difference_list_test_tmp.append(exp_result_tmp["mean_correct_test"] - exp_result_tmp["mean_incorrect_test"])
                difference_list_train_tmp.append(exp_result_tmp["mean_correct_train"] - exp_result_tmp["mean_incorrect_train"])

                AP_dict_tmp['mean_difference_valid'].append(exp_result_tmp["mean_correct_valid"] - exp_result_tmp["mean_incorrect_valid"])
                # AP_dict_tmp['mean_difference_train'].append(exp_result_tmp["mean_correct_train"] - exp_result_tmp["mean_incorrect_train"])
                
                noise_variance_list_tmp.append(exp_result_tmp["hyperparameter"][-1])
                signal_noise_ratio_list_tmp.append((exp_result_tmp["hyperparameter"][1]+exp_result_tmp["hyperparameter"][3])/exp_result_tmp["hyperparameter"][-1])
                
            print("AP_list_test_tmp for {}: {}".format(framework_variant, AP_list_tmp))
            #print("difference_test for {}: {}".format(framework_variant, difference_list_test_tmp))
            #print("difference_train for {}: {}".format(framework_variant, difference_list_train_tmp))
            #print("noise_variance_list_tmp for {}: {}".format(framework_variant, noise_variance_list_tmp))
            #print("signal_noise_ratio_list_tmp for {}: {}".format(framework_variant, signal_noise_ratio_list_tmp))

            AP_df_tmp = pd.DataFrame.from_dict(AP_dict_tmp)
            AP_df_tmp.sort_values(by='mean_difference_valid', inplace=True, ascending=False)

            print(f'Saving summary results for each trial to: {dataset_name}_summary.csv')
            AP_df_tmp[:3].to_csv(os.path.join('Results', f'{dataset_name}_summary_top3.csv'), index=False)
            
            print( AP_df_tmp.to_string(formatters={"fname": lambda pth: str(Path(pth).name)}), end='\n\n')
            print( AP_df_tmp[:3].to_string(formatters={"fname": lambda pth: str(Path(pth).name)}), end='\n\n')
            
            # AP_list.append(np.mean(np.sort(AP_list_tmp, axis=None)[-top_num:]))
            AP_list.append( np.mean(np.sort(AP_df_tmp[metric_name], axis=None)[-top_num:]) )
            AP_dict[metric_name].append( np.mean(np.sort(AP_df_tmp[metric_name], axis=None)[-top_num:]) )
            
        AP_df = pd.DataFrame.from_dict(AP_dict)
        print("AP_list_top{} for {}: {}".format(top_num, framework_variant, AP_df[metric_name].values))
        #print("AP_list_top{} mean for {}: {}".format(top_num, framework_variant, np.mean(AP_list)))

    return AP_df[metric_name]

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

def AP_class_difference(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name):
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
        test_check = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
        softmax_predictions = softmax(exp_result["test_NN_predictions"], axis=1)
        softmax_predictions_sorted = np.sort(softmax_predictions)
        class_diff = softmax_predictions_sorted[:,-1] - softmax_predictions_sorted[:,-2]
        if metric_name == "AP-error" or metric_name == "AUPR-error":
            y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = -class_diff
        else:
            y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
            y_score_test = class_diff
        if metric_name == "AP-error" or metric_name == "AP-success":
            AP_list.append(average_precision_score(y_true_test, y_score_test))
        elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
            precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
            AP_list.append(auc(recall, precision))
        elif metric_name == "AUROC":
            AP_list.append(roc_auc_score(y_true_test, y_score_test))
        #print("AP_list for class difference: {}".format(AP_list))
    return AP_list


# def AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name):
#     AP_list = []
#     for run in range(RUNS):
#         result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_run{}.pkl'.format(dataset_name, run))
#         with open(result_file_name, 'rb') as result_file:
#             exp_info = pickle.load(result_file)
#         if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
#             continue
#         trial = 0
#         result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
#         with open(result_file_name, 'rb') as result_file:
#             exp_result = pickle.load(result_file)
#         test_check = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
#         result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),SOTA_dir_name,('{}_exp_'+SOTA_algo_name+'_{}_run{}.pkl').format(dataset_name, run))
#         try:
#             with open(result_file_name, 'rb') as result_file:
#                 SOTA_exp_info = pickle.load(result_file)
#         except:
#             print("skip {} run{} for {}".format(dataset_name, run, SOTA_algo_name))
#             continue
#         if metric_name == "AP-error" or metric_name == "AUPR-error":
#             y_true_test = (exp_result["test_labels"]!=np.argmax(exp_result["test_NN_predictions"], axis=1))
#             if SOTA_algo_name == "TrustScore":
#                 y_score_test = -(SOTA_exp_info["trust_score_test"].reshape(-1))
#             else:
#                 y_score_test = -(SOTA_exp_info["moderator_test_NN_predictions"].reshape(-1))
#         else:
#             y_true_test = (exp_result["test_labels"]==np.argmax(exp_result["test_NN_predictions"], axis=1))
#             if SOTA_algo_name == "TrustScore":
#                 y_score_test = SOTA_exp_info["trust_score_test"].reshape(-1)
#             else:
#                 y_score_test = SOTA_exp_info["moderator_test_NN_predictions"].reshape(-1)
#         if metric_name == "AP-error" or metric_name == "AP-success":
#             AP_list.append(average_precision_score(y_true_test, y_score_test))
#         elif metric_name == "AUPR-error" or metric_name == "AUPR-success":
#             precision, recall, thresholds = precision_recall_curve(y_true_test, y_score_test)
#             AP_list.append(auc(recall, precision))
#         elif metric_name == "AUROC":
#             AP_list.append(roc_auc_score(y_true_test, y_score_test))
#         #print("AP_list for {}: {}".format(SOTA_algo_name, AP_list))
#     return AP_list

for dataset_index in range(len(dataset_name_list)):

    dataset_name = dataset_name_list[dataset_index]

    # NN_size = "64+64"
    # layer_width = 64
    RUNS = 1
    dir_name = "Results"

    metric_name_list = ["AP-error", "AP-success", "AUPR-error", "AUPR-success", "AUROC"]
    metric_name = metric_name_list[0]

    # NN_info = NN_size

    print("Showing Results for {}".format(dataset_name))
    print("Showing Results for dir: {}".format(dir_name))

    # acc_list = []
    # for run in range(RUNS):
    #     result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
    #     with open(result_file_name, 'rb') as result_file:
    #         exp_info = pickle.load(result_file)
    #     acc_list.append(exp_info["NN_test_acc"])
    # print("NN_test_acc: {}".format(np.mean(acc_list)))

    AP_list_dict = {}
    AP_mean_dict = {}

    kernel_type = "RBF"
    framework_variant = "GP_inputOnly"
    algo_spec = "moderator_residual_target"
    add_info = ""

    # AP_list_dict[framework_variant+"+"+algo_spec+add_info] = AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    # AP_mean_dict[framework_variant+"+"+algo_spec+add_info] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info])

    # label_max_list = []
    # for run in range(RUNS):
    #     result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
    #     with open(result_file_name, 'rb') as result_file:
    #         exp_info = pickle.load(result_file)
    #     if exp_info["NN_test_acc"] == 1.0 or exp_info["NN_test_acc"] == 0.0:
    #         continue
    #     result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_name,'{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
    #     with open(result_file_name, 'rb') as result_file:
    #         exp_result = pickle.load(result_file)
    #     label_max_list.append(np.max(exp_result["train_labels"]))
    # print("max labels: {}".format(label_max_list))

    # AP_list_dict["class_max"] = AP_class_max(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    # AP_mean_dict["class_max"] = np.mean(AP_list_dict["class_max"])
    # AP_list_dict["class_difference"] = AP_class_difference(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    # AP_mean_dict["class_difference"] = np.mean(AP_list_dict["class_difference"])

    kernel_type = "RBF+RBF"
    framework_variant = "GP_corrected"
    top_num = 3
    # AP_list_dict[framework_variant+"+"+algo_spec+add_info] = AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    # AP_mean_dict[framework_variant+"+"+algo_spec+add_info] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info])
    AP_list_dict[framework_variant+"+"+algo_spec+add_info+"topN"] = AP_calculation_topN(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, top_num, metric_name)
    AP_mean_dict[framework_variant+"+"+algo_spec+add_info+"topN"] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info+"topN"])

    # add_info = "+separate_opt"
    # AP_list_dict[framework_variant+"+"+algo_spec+add_info] = AP_calculation(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, metric_name)
    # AP_mean_dict[framework_variant+"+"+algo_spec+add_info] = np.mean(AP_list_dict[framework_variant+"+"+algo_spec+add_info])

    # SOTA_dir_name = "Results"
    # SOTA_algo_name = "CondifNet"
    # AP_list_dict[SOTA_algo_name] = AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name)
    # AP_mean_dict[SOTA_algo_name] = np.mean(AP_list_dict[SOTA_algo_name])
    # SOTA_algo_name = "Introspection-Net"
    # AP_list_dict[SOTA_algo_name] = AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name)
    # AP_mean_dict[SOTA_algo_name] = np.mean(AP_list_dict[SOTA_algo_name])
    # SOTA_dir_name = "Results"
    # SOTA_algo_name = "TrustScore"
    # AP_list_dict[SOTA_algo_name] = AP_SOTA(dataset_name, framework_variant, kernel_type, algo_spec, add_info, RUNS, dir_name, SOTA_dir_name, SOTA_algo_name, metric_name)
    # AP_mean_dict[SOTA_algo_name] = np.mean(AP_list_dict[SOTA_algo_name])

    # info = "All_residual_target_multirun"
    # AP_list_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Statistics','{}_{}_{}.pkl'.format(metric_name, dataset_name, info))
    # with open(AP_list_file_name, 'wb') as result_file:
    #     pickle.dump(AP_list_dict, result_file)
    print(AP_mean_dict)
