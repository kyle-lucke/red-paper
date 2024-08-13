"""
Copyright (C) 2022 Cognizant Digital Business, Evolutionary AI. All Rights Reserved.
Issued under the Academic Public License.
You can be released from the terms, and requirements of the Academic public license by purchasing a commercial license.
"""
from __future__ import absolute_import, division, print_function

import pandas as pd

import tensorflow as tf

import gpflow

from sklearn.metrics import mean_absolute_error

import os
import numpy as np
import time
from scipy.io import arff

# file that contains functions to run RIO variants

def RIO_MRBF_multiple_running_computer_vision(framework_variant,
                                              kernel_type,
                                              
                                              normed_train_data,
                                              normed_valid_data,
                                              normed_test_data,

                                              train_labels,
                                              valid_labels,
                                              test_labels,
                                              
                                              train_NN_predictions,
                                              valid_NN_predictions,
                                              test_NN_predictions,

                                              train_NN_predictions_all,
                                              valid_NN_predictions_all,
                                              test_NN_predictions_all,

                                              M, use_ard, scale_array, separate_opt, batch_size):

    print('!!!!! FIXME !!!!!')
    print("REPLACE 'TEST' WITH 'VALID'")
    print('!!!!! FIXME !!!!!', end='\n\n')

    print('!!!!! FIXME !!!!!')
    print("ADD TEST DATA'")
    print('!!!!! FIXME !!!!!', end='\n\n')
        
    train_NN_errors = (train_labels - train_NN_predictions)
    
    n_train, channels, w, h = normed_train_data.shape
    n_valid, channels, w, h = normed_valid_data.shape
    n_test, channels, w, h = normed_test_data.shape
    
    combined_train_data = normed_train_data.copy().reshape(n_train, -1)
    combined_valid_data = normed_valid_data.copy().reshape(n_valid, -1)
    combined_test_data = normed_test_data.copy().reshape(n_test, -1)

    # input_dimension = len(normed_train_data.columns)
    input_dimension = np.prod(normed_train_data.shape[1:])
    output_dimension = len(train_NN_predictions_all[0])
    
    print(f'input dimension: {input_dimension}')
    print(f'output dimension: {output_dimension}', end='\n\n')
    
    # append NN predictions to end of training/validation data 
    combined_train_data = np.hstack([combined_train_data, train_NN_predictions_all])
    combined_valid_data = np.hstack([combined_valid_data, valid_NN_predictions_all])
    combined_test_data = np.hstack([combined_test_data, test_NN_predictions_all])

    combined_train_data = combined_train_data.astype(np.float64)
    combined_valid_data = combined_valid_data.astype(np.float64)
    combined_test_data = combined_test_data.astype(np.float64)
    
    print(combined_train_data.shape)
    print(combined_valid_data.shape)
    print(combined_test_data.shape)
    
    Z = combined_train_data[:M, :].copy()

    print('FIXME: using 10 iters')
    scipy_options = dict(maxiter=10, disp=False)
    track_loss_history = True
    
    print(Z.shape)
    
    time_checkpoint1 = time.time()

    # select kernel
    if kernel_type == "RBF+RBF":
        print('\n\nselect kernel type RBF+RBF\n\n')

        input_kernel = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension),
            lengthscales=np.ones(input_dimension, dtype=np.float32)
        )
        
        output_kernel = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension, input_dimension + output_dimension),
            lengthscales=np.ones(output_dimension, dtype=np.float32)
        )
        
        k = input_kernel + output_kernel
        
    elif kernel_type == "RBF":
        print('\n\nselect kernel type RBF\n\n')
        
        k = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension),
            lengthscales=np.ones(input_dimension, dtype=np.float32)
        )
        
    elif kernel_type == "RBFY":
        print('\n\nselect kernel type RBFY\n\n')
        
        k = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension, input_dimension + output_dimension),
            lengthscales=np.ones(input_dimension, dtype=np.float32)
        )

        
    # select model / framework variant
    if (framework_variant == "GP_corrected" or
        framework_variant == "GP_corrected_inputOnly" or
        framework_variant == "GP_corrected_outputOnly"):

        X = combined_train_data
        Y = train_NN_errors.reshape(-1, 1)
        
        m = gpflow.models.svgp.SVGP_deprecated(kernel=k,
                                               likelihood=gpflow.likelihoods.Gaussian(),
                                               inducing_variable=Z, num_latent_gps=Y.shape[1])
        
    elif (framework_variant == "GP" or
         framework_variant == "GP_inputOnly" or
         framework_variant == "GP_outputOnly"):

        X = combined_train_data
        Y = train_labels.reshape(-1, 1)
        
        # In gpflow version 2.0, SVGP ported to SVGP_deprecated
        m = gpflow.models.svgp.SVGP_deprecated(kernel=k,
                                               likelihood=gpflow.likelihoods.Gaussian(),
                                               inducing_variable=Z, num_latent_gps=Y.shape[1])

    ############################ hyperparameter optimization #####################################

    print(f'X dtype: {X.dtype}')
    print(f'Y dtype: {Y.dtype}')
    
    # Option 1: optimize hyperparameters for each RBF kernel seperately
    if kernel_type == "RBF+RBF" and separate_opt:
        hyperparameter = (m.kernel.kernels[0].lengthscales.numpy(),
                          m.kernel.kernels[0].variance.numpy(),
                          m.kernel.kernels[1].lengthscales.numpy(),
                          m.kernel.kernels[1].variance.numpy(),
                          m.likelihood.variance.numpy())
        
        print(hyperparameter)

        # optimize input kernel (kernel 0)

        # in gpflow V2 have to use assign to assign Parameter values,
        # and set_trainable to change trainable status

        # m.kernel.kernels[1].variance = 0.0
        m.kernel.kernels[1].variance.assign(1e-8)

        # m.kernel.kernels[1].variance.trainable = False
        # m.kernel.kernels[1].lengthscales.trainable = False 
        gpflow.set_trainable(m.kernel.kernels[1].variance, False)
        gpflow.set_trainable(m.kernel.kernels[1].lengthscales, False)

        # m.kernel.kernels[0].variance = np.random.rand()
        # m.kernel.kernels[0].lengthscales = np.random.rand(len(m.kernel.kernels[0].lengthscales.numpy())) * 10.0
        m.kernel.kernels[0].variance.assign(np.random.rand())
        
        m.kernel.kernels[0].lengthscales.assign(
            np.random.rand(len(m.kernel.kernels[0].lengthscales.numpy())) * 10.0
        )

        # have to call numpy() instead of value for gpflow V2
        hyperparameter = (m.kernel.kernels[0].lengthscales.numpy(),
                          m.kernel.kernels[0].variance.numpy(),
                          m.kernel.kernels[1].lengthscales.numpy(),
                          m.kernel.kernels[1].variance.numpy(),
                          m.likelihood.variance.numpy())
      
        print(hyperparameter)

        # In gpflow V 2.0, the way Scipy optimizer is used changed:
        # opt = gpflow.train.ScipyOptimizer()
        # num_optimizer_iter = opt.minimize(m)

        opt = gpflow.optimizers.Scipy()
        num_optimizer_iter = opt.minimize(
            m.training_loss_closure(data=(X, Y)),
            variables=m.trainable_variables,
            track_loss_history=track_loss_history,
            options=scipy_options
        )
        
        hyperparameter = (m.kernel.kernels[0].lengthscales.numpy(),
                          m.kernel.kernels[0].variance.numpy(),
                          m.kernel.kernels[1].lengthscales.numpy(),
                          m.kernel.kernels[1].variance.numpy(),
                          m.likelihood.variance.numpy())
        
        print(hyperparameter)

        # optimize output kernel (kernel 1)

        # m.kernel.kernels[1].variance = np.random.rand()
        # m.kernel.kernels[1].lengthscales = np.random.rand(len(m.kernel.kernels[1].lengthscales.value)) * 10.0
        
        m.kernel.kernels[1].variance.assign(np.random.rand())
        m.kernel.kernels[1].lengthscales.assign(
            np.random.rand(len(m.kernel.kernels[1].lengthscales.numpy())) * 10.0
        )
        
        # m.kernel.kernels[1].variance.trainable = True # FIXME 
        # m.kernel.kernels[1].lengthscales.trainable = True # FIXME

        gpflow.set_trainable(m.kernel.kernels[1].variance, True)
        gpflow.set_trainable(m.kernel.kernels[1].lengthscales, True)
        
        hyperparameter = (m.kernel.kernels[0].lengthscales.numpy(),
                          m.kernel.kernels[0].variance.numpy(),
                          m.kernel.kernels[1].lengthscales.numpy(),
                          m.kernel.kernels[1].variance.numpy(),
                          m.likelihood.variance.numpy())
        
        print(hyperparameter)

        # In gpflow V 2.0, the way Scipy optimizer is used changed:
        # opt = gpflow.train.ScipyOptimizer()
        # num_optimizer_iter = opt.minimize(m)

        opt = gpflow.optimizers.Scipy()
        num_optimizer_iter = opt.minimize(
            m.training_loss_closure(data=(X, Y)),
            variables=m.trainable_variables,
            track_loss_history=track_loss_history,
            options=scipy_options
        )
        
        hyperparameter = (m.kernel.kernels[0].lengthscales.numpy(),
                          m.kernel.kernels[0].variance.numpy(),
                          m.kernel.kernels[1].lengthscales.numpy(),
                          m.kernel.kernels[1].variance.numpy(),
                          m.likelihood.variance.numpy())
        
        print(hyperparameter)


    # Option 2: optimize kernels together
    elif kernel_type == "RBF+RBF":

        print(m.kernel.kernels[1].lengthscales.numpy())
                
        m.kernel.kernels[1].variance.assign(np.random.rand())
        m.kernel.kernels[1].lengthscales.assign(np.random.rand(len(m.kernel.kernels[1].lengthscales.numpy())) * 10.0 )
        
        m.kernel.kernels[0].variance.assign(np.random.rand()) 
        m.kernel.kernels[0].lengthscales.assign( np.random.rand(len(m.kernel.kernels[0].lengthscales.numpy())) * 10.0 )

        # print(m.kernel.kernels[1].lengthscales)
        
        # print(f'k1 ard?={m.kernel.kernels[0].ard}')
        # print(f'k1 ard?={m.kernel.kernels[1].ard}')

        # In gpflow V 2.0, the way Scipy optimizer is used changed:        
        # opt = gpflow.train.ScipyOptimizer()
        # num_optimizer_iter = opt.minimize(m)

        opt = gpflow.optimizers.Scipy()
        num_optimizer_iter = opt.minimize(
            m.training_loss_closure(data=(X, Y)),
            variables=m.trainable_variables,
            track_loss_history=track_loss_history,
            options=scipy_options
        )
        
    else:
        # In gpflow V 2.0, the way Scipy optimizer is used changed:
        # opt = gpflow.train.ScipyOptimizer()
        # num_optimizer_iter = opt.minimize(m)

        opt = gpflow.optimizers.Scipy()
        num_optimizer_iter = opt.minimize(
            m.training_loss_closure(data=(X, Y)),
            variables=m.trainable_variables,
            track_loss_history=track_loss_history,
            options=scipy_options
        )

    # RBF+RBF has 2 kernels, need to get var/lengthscales for each kernel.
    if kernel_type == "RBF+RBF":
        hyperparameter = (m.kernel.kernels[0].lengthscales.numpy(),
                          m.kernel.kernels[0].variance.numpy(),
                          m.kernel.kernels[1].lengthscales.numpy(),
                          m.kernel.kernels[1].variance.numpy(),
                          m.likelihood.variance.numpy())
    # single kernel: 
    else:
        hyperparameter = (m.kernel.lengthscales.numpy(),
                          m.kernel.variance.numpy(),
                          m.likelihood.variance.numpy())

    mean_valid, var_valid = m.predict_y(combined_valid_data)
    mean_test, var_test = m.predict_y(combined_test_data)
    
    # mean, var returned as gpflow Parameter object, reassign as numpy
    # array for further analysis
    mean_valid, var_valid = mean_valid.numpy(), var_valid.numpy()
    mean_test, var_test = mean_test.numpy(), var_test.numpy()
        
    time_checkpoint2 = time.time()
    computation_time = time_checkpoint2-time_checkpoint1
    print(f"computation_time_{framework_variant}: {time_checkpoint2-time_checkpoint1}")
    
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly":
        valid_final_predictions = valid_NN_predictions + mean_valid.reshape(-1)
        test_final_predictions = test_NN_predictions + mean_test.reshape(-1)

    elif framework_variant == "GP" or framework_variant == "GP_inputOnly" or framework_variant == "GP_outputOnly":
        valid_final_predictions = mean_valid.reshape(-1)
        test_final_predictions = mean_test.reshape(-1)

    MAE_valid = mean_absolute_error(valid_labels, valid_final_predictions)
    print(f"valid mae after {framework_variant}: {MAE_valid}")

    MAE_test = mean_absolute_error(test_labels, test_final_predictions)
    print(f"test mae after {framework_variant}: {MAE_test}")
    
    # count number of points within 95 pct interval
    num_within_interval = 0
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.96 * np.sqrt(var_test.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.96 * np.sqrt(var_test.reshape(-1)[i]):
            num_within_interval += 1
            
    PCT_within95Interval = float(num_within_interval)/len(test_labels)
    print(f"percentage of test points within 95 percent confidence interval ({framework_variant}): {PCT_within95Interval}")

    # count number of points within 90 pct interval
    num_within_interval = 0
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.65 * np.sqrt(var_test.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.65 * np.sqrt(var_test.reshape(-1)[i]):
            num_within_interval += 1
            
    PCT_within90Interval = float(num_within_interval)/len(test_labels)
    print(f"percentage of test points within 90 percent confidence interval ({framework_variant}): {PCT_within90Interval}")
    num_within_interval = 0

    # count number of points within 68 pct interval
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.0 * np.sqrt(var_test.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.0 * np.sqrt(var_test.reshape(-1)[i]):
            num_within_interval += 1
            
    PCT_within68Interval = float(num_within_interval)/len(test_labels)
    print(f"percentage of test points within 68 percent confidence interval ({framework_variant}): {PCT_within68Interval}")

    mean_valid, var_valid = mean_valid.reshape(-1), var_valid.reshape(-1)
    mean_test, var_test = mean_test.reshape(-1), var_test.reshape(-1)

    print(hyperparameter)

    # get mean/variance for training data
    # mean_train, var_train = m.predict_y(combined_train_data)

    # mean_train, var_train = mean_train.numpy(), var_train.numpy()
    
    # mean_train, var_train = mean_train.reshape(-1), var_train.reshape(-1)

    del m
    del opt
    
    return dict(MAE_test=MAE_test,
                MAE_valid=MAE_valid,
                
                PCT_within95Interval=PCT_within95Interval,
                PCT_within90Interval=PCT_within90Interval,
                PCT_within68Interval=PCT_within68Interval,
                
                mean_valid=mean_valid,
                var_valid=var_valid,

                mean_test=mean_test,
                var_test=var_test,

                # computation_time=computation_time,
                hyperparameter=hyperparameter,
                # num_optimizer_iter=num_optimizer_iter,

                # mean_train=mean_train,
                # var_train=var_train,
                )

