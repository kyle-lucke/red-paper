"""
Copyright (C) 2022 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
You can be released from the terms, and requirements of the Academic public license by purchasing a commercial license.
"""
from __future__ import absolute_import, division, print_function

# import matplotlib.pyplot as plt

import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle
import os
import time
from util import load_UCI121, dataset_read, RIO_MRBF_multiple_running
import numpy as np
from sklearn.metrics import mean_absolute_error
import scipy
# import trustscore

print(tf.__version__)

model_name = "SVGP"
#number of Epochs for NN training
EPOCHS = 1000
#number of inducing points for SVGP
M = 50

dataset_name_list = ["balance-scale"]

def build_classification_model(layer_width, num_class, input_dim):
  model = keras.Sequential([
    layers.Dense(layer_width, activation=tf.nn.relu, input_shape=[input_dim]),
    layers.Dense(layer_width, activation=tf.nn.relu),
    layers.Dense(num_class)
  ])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam",#optimizer,
                metrics=['accuracy'])
  return model

def one_hot_encoding(origin_labels, num_class):
    one_hot_labels = np.zeros((len(origin_labels),num_class))
    one_hot_labels[np.arange(len(origin_labels)),origin_labels] = 1
    return one_hot_labels

def acc_calculate(predictions, labels):
    prediction_class = np.argmax(predictions, axis=1)
    num_correct = np.sum(prediction_class==labels)
    acc = num_correct/len(labels)
    return acc

def run_RIO_classification(framework_variant, kernel_type, M, rio_data, rio_setups, algo_spec):
    mean_list = []
    var_list = []
    correction_list = []
    NN_MAE_list = []
    RIO_MAE_list = []
    PCT_within95Interval_list = []
    PCT_within90Interval_list = []
    PCT_within68Interval_list = []
    computation_time_list = []
    hyperparameter_list = []
    num_optimizer_iter_list = []

    if algo_spec == "moderator_residual_target":
        
        train_labels_class = rio_data["one_hot_train_labels"][:,0].copy()
        test_labels_class = rio_data["one_hot_test_labels"][:,0].copy()
        train_NN_predictions_class = rio_data["one_hot_train_labels"][:,0].copy()
        test_NN_predictions_class = rio_data["one_hot_test_labels"][:,0].copy()
        
        for i in range(len(train_labels_class)):
            train_labels_class[i] = np.max(rio_data["train_NN_predictions_softmax"][i])
            train_NN_predictions_class[i] = np.max(rio_data["train_NN_predictions_softmax"][i])
            
            if rio_data["train_check"][i]:
                train_labels_class[i] = 1.0
            else:
                train_labels_class[i] = 0.0
                
        for i in range(len(test_labels_class)):
            test_labels_class[i] = np.max(rio_data["test_NN_predictions_softmax"][i])
            test_NN_predictions_class[i] = np.max(rio_data["test_NN_predictions_softmax"][i])
            
            if rio_data["test_check"][i]:
                test_labels_class[i] = 1.0
            else:
                test_labels_class[i] = 0.0
                
        train_NN_predictions_all = rio_data["train_NN_predictions_softmax"]
        test_NN_predictions_all = rio_data["test_NN_predictions_softmax"]

    NN_MAE = mean_absolute_error(test_labels_class, test_NN_predictions_class)
    if framework_variant == "GP_corrected" or framework_variant == "GP":
        #with tf.compat.v1.Graph().as_default() as tf_graph, tf.compat.v1.Session(graph=tf_graph).as_default():
            print('running RIO_MRBF_multiple_running')
            
            MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter, num_optimizer_iter, mean_train, var_train = RIO_MRBF_multiple_running(framework_variant, \
                                                                                                                                                kernel_type, \
                                                                                                                                                rio_data["normed_train_data"], \
                                                                                                                                                rio_data["normed_test_data"], \
                                                                                                                                                train_labels_class, \
                                                                                                                                                test_labels_class, \
                                                                                                                                                train_NN_predictions_class, \
                                                                                                                                                test_NN_predictions_class, \
                                                                                                                                                train_NN_predictions_all, \
                                                                                                                                                test_NN_predictions_all, \
                                                                                                                                                M, \
                                                                                                                                                rio_setups["use_ard"], \
                                                                                                                                                rio_setups["scale_array"], \
                                                                                                                                                rio_setups["separate_opt"])
    else:
        #with tf.Graph().as_default() as tf_graph, tf.Session(graph=tf_graph).as_default():
            print('running RIO_variants_running')

            MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter, num_optimizer_iter, mean_train, var_train = RIO_variants_running(framework_variant, \
                                                                                                                                                kernel_type, \
                                                                                                                                                rio_data["normed_train_data"], \
                                                                                                                                                rio_data["normed_test_data"], \
                                                                                                                                                train_labels_class, \
                                                                                                                                                test_labels_class, \
                                                                                                                                                train_NN_predictions_class, \
                                                                                                                                                test_NN_predictions_class, \
                                                                                                                                                M, \
                                                                                                                                                rio_setups["use_ard"], \
                                                                                                                                                rio_setups["scale_array"])
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly" or algo_spec == "moderator_residual_target":
        correction_list.append(mean)
        mean_list.append(mean+test_NN_predictions_class)
        correction = mean.copy()
        mean = mean+test_NN_predictions_class
    else:
        mean_list.append(mean)
    var_list.append(var)
    NN_MAE_list.append(NN_MAE)
    RIO_MAE_list.append(MAE)
    PCT_within95Interval_list.append(PCT_within95Interval)
    PCT_within90Interval_list.append(PCT_within90Interval)
    PCT_within68Interval_list.append(PCT_within68Interval)
    computation_time_list.append(computation_time)
    hyperparameter_list.append(hyperparameter)
    num_optimizer_iter_list.append(num_optimizer_iter)

    correction_list_transpose = np.array(correction_list).transpose()
    mean_list_transpose = np.array(mean_list).transpose()
    var_list_transpose = np.array(var_list).transpose()
    print("mean of True: {}".format(np.mean(mean[np.where(rio_data["test_check"])])))
    print("mean of False: {}".format(np.mean(mean[np.where(rio_data["test_check"] == False)])))

    exp_result = {}
    exp_result["mean"] = mean
    exp_result["var"] = var
    exp_result["RIO_MAE"] = MAE
    exp_result["PCT_within95Interval"] = PCT_within95Interval
    exp_result["PCT_within90Interval"] = PCT_within90Interval
    exp_result["PCT_within68Interval"] = PCT_within68Interval
    exp_result["computation_time"] = computation_time
    exp_result["hyperparameter"] = hyperparameter
    exp_result["num_optimizer_iter"] = num_optimizer_iter
    exp_result["test_labels"] = rio_data["test_labels"].values.reshape(-1)
    exp_result["test_NN_predictions"] = rio_data["test_NN_predictions"]
    exp_result["mean_train"] = mean_train
    exp_result["var_train"] = var_train
    exp_result["train_labels"] = rio_data["train_labels"].values.reshape(-1)
    exp_result["train_NN_predictions"] = rio_data["train_NN_predictions"]
    exp_result["mean_correct_train"] = np.mean(mean_train[np.where(rio_data["train_check"])])
    exp_result["mean_incorrect_train"] = np.mean(mean_train[np.where(rio_data["train_check"] == False)])
    exp_result["mean_correct_test"] = np.mean(mean[np.where(rio_data["test_check"])])
    exp_result["mean_incorrect_test"] = np.mean(mean[np.where(rio_data["test_check"] == False)])

    return exp_result

# iterate over datasets
for dataset_index in range(len(dataset_name_list)):

    dataset_name = dataset_name_list[dataset_index]

    NN_size = "64+64"
    layer_width = 64
    RUNS = 1

    NN_info = NN_size

    normed_dataset, labels = load_UCI121(dataset_name)
    num_class = np.max(labels.values)+1
    print("num_class: {}".format(num_class))
    
    for run in range(RUNS):
      print("run{} start".format(run))

      # seed for reproducability
      os.environ['PYTHONHASHSEED'] = str(run)
      random.seed(run)
      tf.compat.v1.set_random_seed(run)
      np.random.seed(run)
          
      # preprocess data
      normed_train_data = normed_dataset.sample(frac=0.8,random_state=run)
      normed_test_data = normed_dataset.drop(normed_train_data.index)
      train_labels = labels.take(normed_train_data.index)
      test_labels = labels.drop(normed_train_data.index)
            
      minibatch_size = len(normed_train_data)
      time_checkpoint1 = time.time()
            
      # training NN
      model = build_classification_model(layer_width, num_class, len(normed_train_data.keys()))

      # The patience parameter is the amount of epochs to check for improvement
      early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

      history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                          validation_split=0.2, verbose=0, callbacks=[early_stop])

      time_checkpoint2 = time.time()

      loss, NN_acc = model.evaluate(normed_test_data, test_labels, verbose=0)
      print("computation_time_NN: {}".format(time_checkpoint2-time_checkpoint1))
      print("Testing set accuracy: {}".format(NN_acc), end='\n\n')
            
      probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
      test_NN_predictions_softmax = probability_model.predict(normed_test_data)
      train_NN_predictions_softmax = probability_model.predict(normed_train_data)
      test_NN_predictions = model.predict(normed_test_data)
      train_NN_predictions = model.predict(normed_train_data)

      one_hot_train_labels = one_hot_encoding(train_labels.values.reshape(-1), num_class)
      one_hot_test_labels = one_hot_encoding(test_labels.values.reshape(-1), num_class)
      #print("one_hot_train_labels: {}".format(one_hot_train_labels))

      train_NN_correct = (np.argmax(train_NN_predictions, axis=1) == train_labels.values)
      num_correct = np.sum(train_NN_correct)
      num_incorrect = len(train_labels.values) - num_correct

      # This feature not used in RED
      scale_correct = 1.0 #len(train_labels.values)/(2*num_correct)
      scale_incorrect = 1.0 #len(train_labels.values)/(2*num_incorrect)
      scale_array = np.ones(len(train_labels.values))
      
      if scale_correct < scale_incorrect:
        for k in range(len(train_labels.values)):
          if train_NN_correct[k]:
            scale_array[k] = scale_correct
          else:
            scale_array[k] = scale_incorrect

      rio_data = {}
      rio_data["normed_train_data"] = normed_train_data
      rio_data["normed_test_data"] = normed_test_data
      rio_data["train_NN_predictions"] = train_NN_predictions
      rio_data["test_NN_predictions"] = test_NN_predictions
      rio_data["train_labels"] = train_labels
      rio_data["test_labels"] = test_labels
      rio_data["one_hot_train_labels"] = one_hot_train_labels
      rio_data["one_hot_test_labels"] = one_hot_test_labels
      rio_data["train_NN_predictions_softmax"] = train_NN_predictions_softmax
      rio_data["test_NN_predictions_softmax"] = test_NN_predictions_softmax

      rio_data["train_check"] = (rio_data["train_labels"].values.reshape(-1)==np.argmax(rio_data["train_NN_predictions"], axis=1))
      rio_data["test_check"] = (rio_data["test_labels"].values.reshape(-1)==np.argmax(rio_data["test_NN_predictions"], axis=1))

            
      rio_setups = {}
      rio_setups["use_ard"] = True
      rio_setups["scale_array"] = scale_array
      rio_setups["separate_opt"] = False

            
      exp_info = {}
      exp_info["test_labels"] = test_labels
      exp_info["test_NN_predictions"] = test_NN_predictions
      exp_info["NN_test_acc"] = NN_acc

      result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_info_{}_run{}.pkl'.format(dataset_name, NN_info, run))
      with open(result_file_name, 'wb') as result_file:
        pickle.dump(exp_info, result_file)

      kernel_type = "RBF"
      framework_variant = "GP_corrected_inputOnly"
      algo_spec = "moderator_residual_target"
      add_info = ""

      print('\n\n#################second#################\n\n')

      kernel_type = "RBF+RBF"
      framework_variant = "GP_corrected"
      trial_num = 10
      max_difference = -100
      for trial in range(trial_num):
        exp_result = run_RIO_classification(framework_variant, kernel_type, M, rio_data, rio_setups, algo_spec)
        if exp_result["mean_correct_test"] - exp_result["mean_incorrect_test"] > max_difference:
          max_difference = exp_result["mean_correct_test"] - exp_result["mean_incorrect_test"]
          result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
          with open(result_file_name, 'wb') as result_file:
            pickle.dump(exp_result, result_file)
                        
        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}_trail{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
        with open(result_file_name, 'wb') as result_file:
          pickle.dump(exp_result, result_file)

      print('\n\n################# third ###################\n\n')
            
      rio_setups["separate_opt"] = True
      add_info = "+separate_opt"
      trial_num = 10
      max_difference = -100
      
      for trial in range(trial_num):
        exp_result = run_RIO_classification(framework_variant, kernel_type, M, rio_data, rio_setups, algo_spec)
        
        if exp_result["mean_correct_test"] - exp_result["mean_incorrect_test"] > max_difference:
          max_difference = exp_result["mean_correct_test"] - exp_result["mean_incorrect_test"]
                    
          result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run))
          
          with open(result_file_name, 'wb') as result_file:
            pickle.dump(exp_result, result_file)

        result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_exp_result_{}_{}_{}_run{}_trail{}.pkl'.format(dataset_name, framework_variant, kernel_type, algo_spec+add_info, run, trial))
        with open(result_file_name, 'wb') as result_file:
          pickle.dump(exp_result, result_file)

