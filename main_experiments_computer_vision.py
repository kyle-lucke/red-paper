"""
Copyright (C) 2022 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
You can be released from the terms, and requirements of the Academic public license by purchasing a commercial license.
"""

import os
import time
import pickle
import random
import argparse

import numpy as np
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from util_computer_vision import RIO_MRBF_multiple_running_computer_vision

# number of inducing points for SVGP
M = 50

BATCH_SIZE = 128

def load_data(input_dir, split):
  inputs = np.load( os.path.join(input_dir, f'{split}_inputs.npy') )
  labels = np.load( os.path.join(input_dir, f'{split}_labels.npy') )

  return inputs, labels

def one_hot_encoding(origin_labels, num_class):
    one_hot_labels = np.zeros((len(origin_labels),num_class))
    one_hot_labels[np.arange(len(origin_labels)),origin_labels] = 1
    return one_hot_labels

def run_RIO_classification(framework_variant, kernel_type, M, rio_data, rio_setups, algo_spec):

    if algo_spec == "moderator_residual_target":
        
        train_labels_class = rio_data["one_hot_train_labels"][:,0].copy()
        valid_labels_class = rio_data["one_hot_valid_labels"][:,0].copy()
        test_labels_class = rio_data["one_hot_test_labels"][:,0].copy()

        train_NN_predictions_class = rio_data["one_hot_train_labels"][:,0].copy()
        valid_NN_predictions_class = rio_data["one_hot_valid_labels"][:,0].copy()
        test_NN_predictions_class = rio_data["one_hot_test_labels"][:,0].copy()
        
        for i in range(len(train_labels_class)):
            train_labels_class[i] = np.max(rio_data["train_NN_predictions_softmax"][i])
            train_NN_predictions_class[i] = np.max(rio_data["train_NN_predictions_softmax"][i])
            
            if rio_data["train_check"][i]:
                train_labels_class[i] = 1.0
            else:
                train_labels_class[i] = 0.0
                
        for i in range(len(valid_labels_class)):
            valid_labels_class[i] = np.max(rio_data["valid_NN_predictions_softmax"][i])
            valid_NN_predictions_class[i] = np.max(rio_data["valid_NN_predictions_softmax"][i])

            # dirac function
            if rio_data["valid_check"][i]:
                valid_labels_class[i] = 1.0
            else:
                valid_labels_class[i] = 0.0

        for i in range(len(test_labels_class)):
            test_labels_class[i] = np.max(rio_data["test_NN_predictions_softmax"][i])
            test_NN_predictions_class[i] = np.max(rio_data["test_NN_predictions_softmax"][i])

            # dirac function
            if rio_data["test_check"][i]:
                test_labels_class[i] = 1.0
            else:
                test_labels_class[i] = 0.0

        train_NN_predictions_all = rio_data["train_NN_predictions_softmax"]
        valid_NN_predictions_all = rio_data["valid_NN_predictions_softmax"]
        test_NN_predictions_all = rio_data["test_NN_predictions_softmax"]
    
    NN_MAE_test = mean_absolute_error(test_labels_class, test_NN_predictions_class)
    if framework_variant == "GP_corrected" or framework_variant == "GP":

      print('running RIO_MRBF_multiple_running_compute_vision')
            
      # MAE_test, MAE_valid, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean_valid, var_valid, mean_test, var_test, computation_time, hyperparameter, num_optimizer_iter, mean_train, var_train, = RIO_MRBF_multiple_running_computer_vision(

      res = RIO_MRBF_multiple_running_computer_vision(
        
        framework_variant,
        kernel_type,
        
        rio_data["normed_train_data"],
        rio_data["normed_valid_data"],
        rio_data["normed_test_data"],
        
        train_labels_class, 
        valid_labels_class,
        test_labels_class, 

        train_NN_predictions_class, 
        valid_NN_predictions_class,
        test_NN_predictions_class, 

        train_NN_predictions_all, 
        valid_NN_predictions_all,
        test_NN_predictions_all, 

        M, 
        rio_setups["use_ard"], 
        rio_setups["scale_array"], 
        rio_setups["separate_opt"],
        BATCH_SIZE
      )
            
    else:
        
      print('running RIO_variants_running')

      # MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter, num_optimizer_iter, mean_train, var_train = RIO_variants_running(
      #   framework_variant,
      #   kernel_type,
              
      #   rio_data["normed_train_data"],
      #   rio_data["normed_valid_data"],

      #   train_labels_class,
      #   valid_labels_class,
              
      #   train_NN_predictions_class,
      #   valid_NN_predictions_class,
              
      #   M,
      #   rio_setups["use_ard"],
      #   rio_setups["scale_array"])

    # apply correction for framework variants
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly" or algo_spec == "moderator_residual_target":

        res['mean_train'] = res['mean_train']+train_NN_predictions_class
        res['mean_valid'] = res['mean_valid']+valid_NN_predictions_class
        res['mean_test'] = res['mean_test']+test_NN_predictions_class

    print("mean of True: {}".format(np.mean(res['mean_test'][np.where(rio_data["test_check"])])))
    print("mean of False: {}".format(np.mean(res['mean_test'][np.where(rio_data["test_check"] == False)])))

    exp_result = {}
        
    exp_result["RIO_MAE_test"] = res['MAE_test']
    exp_result["RIO_MAE_valid"] = res['MAE_valid']
    
    exp_result["PCT_within95Interval"] = res['PCT_within95Interval']
    exp_result["PCT_within90Interval"] = res['PCT_within90Interval']
    exp_result["PCT_within68Interval"] = res['PCT_within68Interval']

    # exp_result["computation_time"] = res['computation_time']
    exp_result["hyperparameter"] = res['hyperparameter']
    # exp_result["num_optimizer_iter"] = res['num_optimizer_iter']

    exp_result["mean_train"] = res['mean_train']
    exp_result["var_train"] = res['var_train']

    exp_result["mean_valid"] = res['mean_valid']
    exp_result["var_valid"] = res['var_valid']
    
    exp_result["mean_test"] = res['mean_test']
    exp_result["var_test"] = res['var_test']
    
    exp_result["train_labels"] = rio_data["train_labels"].reshape(-1)
    exp_result["train_NN_predictions"] = rio_data["train_NN_predictions"]
    
    exp_result["valid_labels"] = rio_data["valid_labels"].reshape(-1)
    exp_result["valid_NN_predictions"] = rio_data["valid_NN_predictions"]
    
    exp_result["test_labels"] = rio_data["test_labels"].reshape(-1)
    exp_result["test_NN_predictions"] = rio_data["test_NN_predictions"]
    
    exp_result["mean_correct_train"] = np.mean(res['mean_train'][np.where(rio_data["train_check"])])
    exp_result["mean_incorrect_train"] = np.mean(res['mean_train'][np.where(rio_data["train_check"] == False)])

    exp_result["mean_correct_valid"] = np.mean(res['mean_valid'][np.where(rio_data["valid_check"])])
    exp_result["mean_incorrect_valid"] = np.mean(res['mean_valid'][np.where(rio_data["valid_check"] == False)])

    exp_result["mean_correct_test"] = np.mean(res['mean_test'][np.where(rio_data["test_check"])])
    exp_result["mean_incorrect_test"] = np.mean(res['mean_test'][np.where(rio_data["test_check"] == False)])
    
    return exp_result

parser = argparse.ArgumentParser()

parser.add_argument('input_dir')
parser.add_argument('base_model', choices=['svhn-cnn', 'cifar10-resnet32'])

args = parser.parse_args()

print('args:')
for k, v in vars(args).items():
  print(f'{k}: {v}')
print()

os.makedirs('Results', exist_ok=True)

RUNS = 1

# load data
normed_train_data, train_labels = load_data(args.input_dir, 'train_meta') 
normed_valid_data, valid_labels = load_data(args.input_dir, 'val')
normed_test_data, test_labels = load_data(args.input_dir, 'test')

train_NN_predictions_softmax = np.load(os.path.join(args.input_dir,
                                                      'train_meta_predictions_softmax.npy'))
  
valid_NN_predictions_softmax = np.load(os.path.join(args.input_dir,
                                                      'val_predictions_softmax.npy'))
  
test_NN_predictions_softmax = np.load(os.path.join(args.input_dir,
                                                      'test_predictions_softmax.npy'))
  
train_NN_predictions = np.load(os.path.join(args.input_dir, 'train_meta_predictions.npy'))
valid_NN_predictions = np.load(os.path.join(args.input_dir, 'val_predictions.npy'))
test_NN_predictions = np.load(os.path.join(args.input_dir, 'test_predictions.npy'))

# print('!!!!!! FIXME !!!!!')
# print('USING SUBSET DATA')

# normed_train_data, train_labels = normed_train_data[:1000], train_labels[:1000]
# normed_valid_data, valid_labels = normed_valid_data[:1000], valid_labels[:1000]
# normed_test_data, test_labels = normed_test_data[:1000], test_labels[:1000]

# train_NN_predictions_softmax = train_NN_predictions_softmax[:1000]
  
# valid_NN_predictions_softmax = valid_NN_predictions_softmax[:1000]
  
# test_NN_predictions_softmax = test_NN_predictions_softmax[:1000]
  
# train_NN_predictions = train_NN_predictions[:1000]
# valid_NN_predictions = valid_NN_predictions[:1000]
# test_NN_predictions = test_NN_predictions[:1000] 
 
# print('!!!!!! FIXME !!!!!')

valid_acc = np.mean(np.argmax(valid_NN_predictions, -1) == valid_labels)
test_acc = np.mean(np.argmax(test_NN_predictions, -1) == test_labels)

print('val acc:', valid_acc)
print('test acc:', test_acc, end='\n\n', flush=True)

num_class = np.max(train_labels)+1

print('number of classes:', num_class, end='\n\n')

for run in range(RUNS):
  print("run{} start".format(run))

  # seed for reproducability
  tf.config.experimental.enable_op_determinism()
  os.environ['PYTHONHASHSEED'] = str(run)
  keras.utils.set_random_seed(run)
    
  one_hot_train_labels = one_hot_encoding(train_labels.reshape(-1), num_class)
  one_hot_valid_labels = one_hot_encoding(valid_labels.reshape(-1), num_class)
  one_hot_test_labels = one_hot_encoding(test_labels.reshape(-1), num_class)

  #print("one_hot_train_labels: {}".format(one_hot_train_labels))
  
  train_NN_correct = (np.argmax(train_NN_predictions, axis=1) == train_labels)
  num_correct = np.sum(train_NN_correct)
  num_incorrect = len(train_labels) - num_correct

  # This feature not used in RED
  scale_correct = 1.0 
  scale_incorrect = 1.0 
  scale_array = np.ones(len(train_labels))
      
  if scale_correct < scale_incorrect:
    for k in range(len(train_labels)):
      if train_NN_correct[k]:
        scale_array[k] = scale_correct
      else:
        scale_array[k] = scale_incorrect

  rio_data = {}
  
  rio_data["normed_train_data"] = normed_train_data
  rio_data["normed_valid_data"] = normed_valid_data
  rio_data["normed_test_data"] = normed_test_data
  
  rio_data["train_NN_predictions"] = train_NN_predictions
  rio_data["valid_NN_predictions"] = valid_NN_predictions
  rio_data["test_NN_predictions"] = test_NN_predictions

  rio_data["train_labels"] = train_labels
  rio_data["valid_labels"] = valid_labels
  rio_data["test_labels"] = test_labels

  rio_data["one_hot_train_labels"] = one_hot_train_labels
  rio_data["one_hot_valid_labels"] = one_hot_valid_labels
  rio_data["one_hot_test_labels"] = one_hot_test_labels

  rio_data["train_NN_predictions_softmax"] = train_NN_predictions_softmax
  rio_data["valid_NN_predictions_softmax"] = valid_NN_predictions_softmax
  rio_data["test_NN_predictions_softmax"] = test_NN_predictions_softmax
  
  rio_data["train_check"] = (rio_data["train_labels"].reshape(-1)==
                             np.argmax(rio_data["train_NN_predictions"], axis=1))

  rio_data["valid_check"] = (rio_data["valid_labels"].reshape(-1)==
                             np.argmax(rio_data["valid_NN_predictions"], axis=1))

  rio_data["test_check"] = (rio_data["test_labels"].reshape(-1)==
                             np.argmax(rio_data["test_NN_predictions"], axis=1))
    
  rio_setups = {}
  rio_setups["use_ard"] = True
  rio_setups["scale_array"] = scale_array
  rio_setups["separate_opt"] = False

  exp_info = {}
  exp_info["valid_labels"] = valid_labels
  exp_info["valid_NN_predictions"] = valid_NN_predictions
  
  exp_info["test_labels"] = test_labels
  exp_info["test_NN_predictions"] = test_NN_predictions
  
  result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_run{}.pkl'.format(args.base_model, run))
  with open(result_file_name, 'wb') as result_file:
    pickle.dump(exp_info, result_file)

  algo_spec = "moderator_residual_target"
  add_info = ""
  
  kernel_type = "RBF+RBF"
  framework_variant = "GP_corrected"
  trial_num = 10
  max_difference = -100
  for trial in range(trial_num):

    # # FIXME: debug
    # if trial > 2:
    #   print(f'WARNING: skipping trial {trial}')
    #   continue

    exp_result = run_RIO_classification(framework_variant, kernel_type, M, rio_data, rio_setups, algo_spec)
    
    if exp_result["mean_correct_valid"] - exp_result["mean_incorrect_valid"] > max_difference:

      print(f'\n\nMax difference improved from: {max_difference} to {exp_result["mean_correct_valid"] - exp_result["mean_incorrect_valid"]}.\nSaving results to {result_file_name}', end='\n\n')

      max_difference = exp_result["mean_correct_valid"] - exp_result["mean_incorrect_valid"]
      
      result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(args.base_model, framework_variant, kernel_type, algo_spec+add_info, run))
      
      with open(result_file_name, 'wb') as result_file:
        pickle.dump(exp_result, result_file)

    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(args.base_model, framework_variant, kernel_type, algo_spec+add_info, run, trial))
    with open(result_file_name, 'wb') as result_file:
      pickle.dump(exp_result, result_file)
  
  
  print('\n\n################# third ###################\n\n')
            
  rio_setups["separate_opt"] = True
  add_info = "+separate_opt"
  trial_num = 10
  max_difference = -100
  
  for trial in range(trial_num):

    # # FIXME: debug
    # if trial > 2:
    #   print(f'WARNING: skipping trial {trial}')
    #   continue
    
    exp_result = run_RIO_classification(framework_variant, kernel_type, M, rio_data,
                                        rio_setups, algo_spec)

    # FIXME: save best, use mean_test as score: 
    if exp_result["mean_correct_valid"] - exp_result["mean_incorrect_valid"] > max_difference:
                          
      result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(args.base_model, framework_variant, kernel_type, algo_spec+add_info, run))

      print(f'\n\nMax difference improved from: {max_difference} to {exp_result["mean_correct_valid"] - exp_result["mean_incorrect_valid"]}.\nSaving results to {result_file_name}', end='\n\n')

      max_difference = exp_result["mean_correct_valid"] - exp_result["mean_incorrect_valid"]
      
      with open(result_file_name, 'wb') as result_file:
        pickle.dump(exp_result, result_file)
        
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}_trial{}.pkl'.format(args.base_model, framework_variant, kernel_type, algo_spec+add_info, run, trial))
    with open(result_file_name, 'wb') as result_file:
      pickle.dump(exp_result, result_file)
      
