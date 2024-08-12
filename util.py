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

# file that contains functions to read dataset and run RIO variants

def load_UCI121(dataset_name):
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'UCI121_data',dataset_name,'{}_py.dat'.format(dataset_name))
    normed_dataset = pd.read_csv(dataset_path, header=None, sep=",")
    label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'UCI121_data',dataset_name,'labels_py.dat')
    labels = pd.read_csv(label_path, header=None).astype(int)
    return normed_dataset.dropna(), labels.dropna()

def dataset_read(dataset_name):
    if dataset_name == "yacht":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','yacht_hydrodynamics.data')
        column_names = ['Longitudinal position of the center of buoyancy','Prismatic coefficient','Length-displacement ratio','Beam-draught ratio','Length-beam ratio','Froude number','Residuary resistance']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, sep=' +', engine='python')
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "ENB_heating":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','ENB2012_data.xlsx')
        raw_dataset = pd.read_excel(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('Y2')
    elif dataset_name == "ENB_cooling":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','ENB2012_data.xlsx')
        raw_dataset = pd.read_excel(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('Y1')
    elif dataset_name == "airfoil_self_noise":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','airfoil_self_noise.dat')
        column_names = ['Frequency','Angle of attack','Chord length','Free-stream velocity','Suction side displacement thickness','sound pressure']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, sep="\t")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "concrete":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Concrete_Data.xls')
        raw_dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "winequality-red":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','winequality-red.csv')
        raw_dataset = pd.read_csv(dataset_path, sep = ';').astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "winequality-white":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','winequality-white.csv')
        raw_dataset = pd.read_csv(dataset_path, sep = ';').astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "CCPP":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Combined_Cycle_Power_Plant.xlsx')
        raw_dataset = pd.read_excel(dataset_path, sheet_name="Sheet1")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "CASP":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','CASP.csv')
        raw_dataset = pd.read_csv(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "SuperConduct":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','SuperConduct.csv')
        raw_dataset = pd.read_csv(dataset_path)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "slice_localization":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','slice_localization_data.csv')
        raw_dataset = pd.read_csv(dataset_path) + 0.01
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop('patientId')
    elif dataset_name == "MSD":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','YearPredictionMSD.txt')
        raw_dataset = pd.read_csv(dataset_path, sep=",", header=None)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset = dataset.astype(float)
    elif dataset_name == "Climate":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','pop_failures.dat')
        raw_dataset = pd.read_csv(dataset_path, sep=' +', engine='python')
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "Bioconcentration":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','Grisoni_et_al_2016_EnvInt88.csv')
        raw_dataset = pd.read_csv(dataset_path, sep="\t")
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "messidor":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','messidor_features.arff')
        data = arff.loadarff(dataset_path)
        raw_dataset = pd.DataFrame(data[0]).astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
    elif dataset_name == "Phishing":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','PhishingData.arff')
        data = arff.loadarff(dataset_path)
        raw_dataset = pd.DataFrame(data[0]).astype(float)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset["Result"] += 1
    elif dataset_name == "yeast":
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets','yeast.data')
        raw_dataset = pd.read_csv(dataset_path, header=None, sep=' +', engine='python')
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset.pop(0)
        class_dic = {}
        count_dic = {}
        for i in range(len(dataset[9].values)):
            if dataset[9].values[i] in class_dic:
                dataset.set_value(i, 9, class_dic[dataset[9].values[i]])
                count_dic[dataset[9].values[i]] += 1
            else:
                class_dic[dataset[9].values[i]] = len(class_dic)
                dataset.set_value(i, 9, len(class_dic)-1)
                count_dic[dataset[9].values[i]] = 1
    return dataset


def RIO_MRBF_multiple_running(framework_variant, kernel_type, normed_train_data,
                              normed_test_data, train_labels, test_labels,
                              train_NN_predictions, test_NN_predictions,
                              train_NN_predictions_all, test_NN_predictions_all, M,
                              use_ard, scale_array, separate_opt):
    
    train_NN_errors = train_labels - train_NN_predictions
    output_dimension = len(train_NN_predictions_all[0])
    combined_train_data = normed_train_data.copy()
    combined_test_data = normed_test_data.copy()

    for i in range(output_dimension):
        combined_train_data['prediction{}'.format(i)] = train_NN_predictions_all[:,i]
        combined_test_data['prediction{}'.format(i)] = test_NN_predictions_all[:,i]

    minibatch_size = len(normed_train_data)
    input_dimension = len(normed_train_data.columns)

    print(f"output_dimension: {output_dimension}")

    Z = combined_train_data.values[:M, :].copy()
    
    time_checkpoint1 = time.time()

    # select kernel
    if kernel_type == "RBF+RBF":
        print('\n\nselect kernel type RBF+RBF\n\n')

        input_kernel = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension),
            lengthscales=np.ones(input_dimension)
        )
        
        output_kernel = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension, input_dimension + output_dimension),
            lengthscales=np.ones(output_dimension)
        )
        
        k = input_kernel + output_kernel
        
    elif kernel_type == "RBF":
        print('\n\nselect kernel type RBF\n\n')
        
        k = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension),
            lengthscales=np.ones(input_dimension)
        )
        
    elif kernel_type == "RBFY":
        print('\n\nselect kernel type RBFY\n\n')
        
        k = gpflow.kernels.SquaredExponential(
            active_dims=range(input_dimension, input_dimension + output_dimension),
            lengthscales=np.ones(input_dimension)
        )


    # select model / framework variant
    if (framework_variant == "GP_corrected" or
        framework_variant == "GP_corrected_inputOnly" or
        framework_variant == "GP_corrected_outputOnly"):

        X = combined_train_data.values
        Y = train_NN_errors.reshape(-1, 1)
        
        m = gpflow.models.svgp.SVGP_deprecated(kernel=k,
                                               likelihood=gpflow.likelihoods.Gaussian(),
                                               inducing_variable=Z, num_latent_gps=Y.shape[1])
        
    elif (framework_variant == "GP" or
         framework_variant == "GP_inputOnly" or
         framework_variant == "GP_outputOnly"):

        X = combined_train_data.values
        Y = train_labels.reshape(-1, 1)

        # In gpflow version 2.0, SVGP ported to SVGP_deprecated
        m = gpflow.models.svgp.SVGP_deprecated(kernel=k,
                                               likelihood=gpflow.likelihoods.Gaussian(),
                                               inducing_variable=Z, num_latent_gps=Y.shape[1])

    ############################ hyperparameter optimization #####################################

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
            options=dict(maxiter=1000)
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
            options=dict(maxiter=1000)
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
            options=dict(maxiter=1000)
        )
        
    else:
        # In gpflow V 2.0, the way Scipy optimizer is used changed:
        # opt = gpflow.train.ScipyOptimizer()
        # num_optimizer_iter = opt.minimize(m)

        opt = gpflow.optimizers.Scipy()
        num_optimizer_iter = opt.minimize(
            m.training_loss_closure(data=(X, Y)),
            variables=m.trainable_variables,
            options=dict(maxiter=1000)
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

    mean, var = m.predict_y(combined_test_data.values)

    # mean, var returned as gpflow Parameter object, reassign as numpy
    # array for further analysis
    mean = mean.numpy()
    var = var.numpy()
    
    time_checkpoint2 = time.time()
    computation_time = time_checkpoint2-time_checkpoint1
    print(f"computation_time_{framework_variant}: {time_checkpoint2-time_checkpoint1}")
    
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly":
        test_final_predictions = test_NN_predictions + mean.reshape(-1)

    elif framework_variant == "GP" or framework_variant == "GP_inputOnly" or framework_variant == "GP_outputOnly":
        test_final_predictions = mean.reshape(-1)

    MAE = mean_absolute_error(test_labels, test_final_predictions)
    print(f"test mae after {framework_variant}: {MAE}")

    # count number of points within 95 pct interval
    num_within_interval = 0
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.96 * np.sqrt(var.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.96 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
            
    PCT_within95Interval = float(num_within_interval)/len(test_labels)
    print(f"percentage of test points within 95 percent confidence interval ({framework_variant}): {PCT_within95Interval}")

    # count number of points within 90 pct interval
    num_within_interval = 0
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.65 * np.sqrt(var.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.65 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
            
    PCT_within90Interval = float(num_within_interval)/len(test_labels)
    print(f"percentage of test points within 90 percent confidence interval ({framework_variant}): {PCT_within90Interval}")
    num_within_interval = 0

    # count number of points within 68 pct interval
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.0 * np.sqrt(var.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.0 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
            
    PCT_within68Interval = float(num_within_interval)/len(test_labels)
    print(f"percentage of test points within 68 percent confidence interval ({framework_variant}): {PCT_within68Interval}")

    mean = mean.reshape(-1)
    var = var.reshape(-1)

    print(hyperparameter)

    # get mean/variance for training data
    mean_train, var_train = m.predict_y(combined_train_data.values)

    mean_train = mean_train.numpy()
    var_train = var_train.numpy()
    
    mean_train = mean_train.reshape(-1)
    var_train = var_train.reshape(-1)

    return MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter, num_optimizer_iter, mean_train, var_train

def RIO_MRBF_running(framework_variant, kernel_type, normed_train_data, normed_test_data, train_labels, test_labels, train_NN_predictions, test_NN_predictions, train_NN_predictions_all, test_NN_predictions_all, M, use_ard, scale_array, separate_opt):
    train_NN_errors = train_labels - train_NN_predictions
    output_dimension = len(train_NN_predictions_all[0])
    combined_train_data = normed_train_data.copy()
    combined_test_data = normed_test_data.copy()
    for i in range(output_dimension):
        combined_train_data['prediction{}'.format(i)] = train_NN_predictions_all[:,i]
        combined_test_data['prediction{}'.format(i)] = test_NN_predictions_all[:,i]
    minibatch_size = len(normed_train_data)
    input_dimension = len(normed_train_data.columns)
    print("output_dimension: {}".format(output_dimension))
    Z = combined_train_data.values[:M, :].copy()
    time_checkpoint1 = time.time()
    if kernel_type == "RBF+RBF":
        k = gpflow.kernels.SquaredExponential(input_dim=input_dimension, active_dims=range(input_dimension), ARD=use_ard) \
            + gpflow.kernels.SquaredExponential(input_dim=output_dimension, active_dims=range(input_dimension, input_dimension + output_dimension), ARD=use_ard)
    elif kernel_type == "RBF":
        k = gpflow.kernels.SquaredExponential(input_dim=input_dimension, active_dims=range(input_dimension), ARD=use_ard)
    elif kernel_type == "RBFY":
        k = gpflow.kernels.SquaredExponential(input_dim=output_dimension, active_dims=range(input_dimension, input_dimension + output_dimension), ARD=use_ard)
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly":
        m = gpflow.models.SVGP(combined_train_data.values, train_NN_errors.reshape(-1,1), kern=k, likelihood=gpflow.likelihoods.Gaussian(), Z=Z)#, minibatch_size=minibatch_size)
    elif framework_variant == "GP" or framework_variant == "GP_inputOnly" or framework_variant == "GP_outputOnly":
        m = gpflow.models.SVGP(combined_train_data.values, train_labels.reshape(-1,1), kern=k, likelihood=gpflow.likelihoods.Gaussian(), Z=Z)#, minibatch_size=minibatch_size)

    if kernel_type == "RBF+RBF" and separate_opt:
        hyperparameter = (m.kern.kernels[0].lengthscales.value, m.kern.kernels[0].variance.value, m.kern.kernels[1].lengthscales.value, m.kern.kernels[1].variance.value, m.likelihood.variance.value)
        print(hyperparameter)
        m.kern.kernels[1].variance = 0.0
        m.kern.kernels[1].variance.trainable = False
        m.kern.kernels[1].lengthscales.trainable = False
        hyperparameter = (m.kern.kernels[0].lengthscales.value, m.kern.kernels[0].variance.value, m.kern.kernels[1].lengthscales.value, m.kern.kernels[1].variance.value, m.likelihood.variance.value)
        print(hyperparameter)
        opt = gpflow.train.ScipyOptimizer()
        num_optimizer_iter = opt.minimize(m)
        hyperparameter = (m.kern.kernels[0].lengthscales.value, m.kern.kernels[0].variance.value, m.kern.kernels[1].lengthscales.value, m.kern.kernels[1].variance.value, m.likelihood.variance.value)
        print(hyperparameter)
        m.kern.kernels[1].variance = 1.0
        m.kern.kernels[1].variance.trainable = True
        m.kern.kernels[1].lengthscales.trainable = True
        hyperparameter = (m.kern.kernels[0].lengthscales.value, m.kern.kernels[0].variance.value, m.kern.kernels[1].lengthscales.value, m.kern.kernels[1].variance.value, m.likelihood.variance.value)
        print(hyperparameter)
        opt = gpflow.train.ScipyOptimizer()
        num_optimizer_iter = opt.minimize(m)
        hyperparameter = (m.kern.kernels[0].lengthscales.value, m.kern.kernels[0].variance.value, m.kern.kernels[1].lengthscales.value, m.kern.kernels[1].variance.value, m.likelihood.variance.value)
        print(hyperparameter)
    else:
        opt = gpflow.train.ScipyOptimizer()
        num_optimizer_iter = opt.minimize(m)

    if kernel_type == "RBF+RBF":
        hyperparameter = (m.kern.kernels[0].lengthscales.value, m.kern.kernels[0].variance.value, m.kern.kernels[1].lengthscales.value, m.kern.kernels[1].variance.value, m.likelihood.variance.value)
    else:
        hyperparameter = (m.kern.lengthscales.value, m.kern.variance.value, m.likelihood.variance.value)
    mean, var = m.predict_y(combined_test_data.values)
    time_checkpoint2 = time.time()
    computation_time = time_checkpoint2-time_checkpoint1
    print("computation_time_{}: {}".format(framework_variant, time_checkpoint2-time_checkpoint1))
    if framework_variant == "GP_corrected" or framework_variant == "GP_corrected_inputOnly" or framework_variant == "GP_corrected_outputOnly":
        test_final_predictions = test_NN_predictions + mean.reshape(-1)
    elif framework_variant == "GP" or framework_variant == "GP_inputOnly" or framework_variant == "GP_outputOnly":
        test_final_predictions = mean.reshape(-1)

    MAE = mean_absolute_error(test_labels, test_final_predictions)
    print("test mae after {}: {}".format(framework_variant, MAE))

    num_within_interval = 0
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.96 * np.sqrt(var.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.96 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
    PCT_within95Interval = float(num_within_interval)/len(test_labels)
    print("percentage of test points within 95 percent confidence interval ({}): {}".format(framework_variant, PCT_within95Interval))
    num_within_interval = 0
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.65 * np.sqrt(var.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.65 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
    PCT_within90Interval = float(num_within_interval)/len(test_labels)
    print("percentage of test points within 90 percent confidence interval ({}): {}".format(framework_variant, PCT_within90Interval))
    num_within_interval = 0
    for i in range(len(test_labels)):
        if test_labels[i] <= test_final_predictions[i] + 1.0 * np.sqrt(var.reshape(-1)[i]) and test_labels[i] >= test_final_predictions[i] - 1.0 * np.sqrt(var.reshape(-1)[i]):
            num_within_interval += 1
    PCT_within68Interval = float(num_within_interval)/len(test_labels)
    print("percentage of test points within 68 percent confidence interval ({}): {}".format(framework_variant, PCT_within68Interval))
    mean = mean.reshape(-1)
    var = var.reshape(-1)
    print(hyperparameter)

    mean_train, var_train = m.predict_y(combined_train_data.values)
    mean_train = mean_train.reshape(-1)
    var_train = var_train.reshape(-1)

    return MAE, PCT_within95Interval, PCT_within90Interval, PCT_within68Interval, mean, var, computation_time, hyperparameter, num_optimizer_iter, mean_train, var_train
