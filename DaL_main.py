import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import os
from random import sample
from utils.general import build_model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression
from utils.general import *
from utils.runHINNPerf import get_HINNPerf_MRE, get_HINNPerf_best_config
from utils.HINNPerf_data_preproc import system_samplesize, seed_generator, DataPreproc
from utils.HINNPerf_args import list_of_param_dicts
from utils.HINNPerf_models import MLPHierarchicalModel
from utils.HINNPerf_model_runner import ModelRunner
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    learning_model = 'DaL-HINNPerf'
    selected_datasets = [10] # menu: 0 - Apache_AllNumeric, 1 - BDBC_AllNumeric, 2 - BDBJ_AllNumeric, 3 - Dune_AllNumeric, 4 - Lrzip, 5 - VP8, 6 - hipacc_AllNumeric, 7 - hsmgp_AllNumeric, 8 - kanzi, 9 - nginx, 10 - sqlite, 11 - x264_AllNumeric
    selected_sizes = [0] # choose from [0, 1, 2, 3, 4]
    save_results = False # save the results
    test_mode = True # if True, disable hyperparameter tuning for a quick test run
    min_samples = 4 # the minimum number of samples per division
    end_run = 30 # N_experiments = end_run-start_run
    start_run = 0 # start from 0
    min_depth = 0 # the minimum depth to select
    max_epoch = 2000 # max_epoch for training the local model
    N_experiments = end_run-start_run
    depth_selection_mode = ['AvgHV'] # 'AvgHV' / 'fixed-1', 'fixed-2', 'fixed-3', 'fixed-4'

    # get all available datasets
    file_names = []
    for home, dirs, files in os.walk('Data/'.format()):
        for filename in files:
            file_names.append(filename)
    file_names.sort()
    dir_datas = ['Data/{}'.format(file_name) for file_name in file_names]
    for temp, temp_file in enumerate(file_names):
        print('{}-{} '.format(temp, temp_file))
    print('\nRuning {}, save_results: {}, test_mode: {}, run{}-{}, depth_selection_mode: {}, selected_sizes: {}, selected_datasets: {}...'.format(learning_model, save_results, test_mode, start_run, end_run, depth_selection_mode, selected_sizes, selected_datasets))

    for mode in depth_selection_mode:
        for dir_data in [dir_datas[temp] for temp in selected_datasets]: # for each dataset
            print('Dataset: ' + dir_data)
            whole_data = load_data(dir_data)
            (N, n) = whole_data.shape
            N_features = n - 1
            non_zero_indexes = get_non_zero_indexes(whole_data) # delete the zero-performance samples
            print('Total sample size: ', len(non_zero_indexes))
            print('N_features: ', N_features)
            print('N_expriments: ', N_experiments)
            subject_system = dir_data.split('/')[1].split('.')[0]
            sample_sizes = get_sample_sizes(subject_system)
            print('Sample sizes: {}'.format(sample_sizes))
            saving_folder = '{}/results/{}'.format(os.getcwd(), subject_system)

            if not os.path.exists(saving_folder):
                print('Creating folder: {}'.format(saving_folder))
                os.makedirs(saving_folder)

            if not mode.startswith('fixed'):
                if len(mode.split('-')) > 1:
                    temp_mode = mode.split('-')[0]
                    error_mode = mode.split('-')[1]
                    reading_file_name = '{}_depths_{}.csv'.format(subject_system, temp_mode)
                    temp_depths = pd.read_csv('{}/selected_depth_{}/{}'.format(os.getcwd(), error_mode, reading_file_name))
                else:
                    reading_file_name = '{}_depths.csv'.format(subject_system)
                    temp_depths = pd.read_csv('{}/selected_depth/{}'.format(os.getcwd(), reading_file_name))
                print('reading {}...'.format(reading_file_name))
            elif mode.startswith('fixed') and len(mode.split('-')) > 1:
                max_depth = int(mode.split('-')[1])
            else:
                max_depth = 1

            for i_size in selected_sizes:
                N_train = sample_sizes[i_size]
                temp_seeds = pd.read_csv('{}/{}'.format(os.getcwd(), 'Seeds.csv'))
                temp_sys_index = list(temp_seeds['System']).index(subject_system)
                if temp_seeds['size{}'.format(i_size + 1)][temp_sys_index] != 'None':
                    seed = int(temp_seeds['size{}'.format(i_size + 1)][temp_sys_index])
                else:
                    print('size{} seed = None'.format(i_size + 1))
                    seed = 1
                print('N_train: {}, Seed: {}'.format(N_train, seed))

                non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)

                if N_train > int(len(non_zero_indexes) * 8 / 10):
                    N_train = int(len(non_zero_indexes) * 8 / 10)
                N_test = (len(non_zero_indexes) - N_train)

                for ne in range(start_run, end_run):
                    if not mode.startswith('fixed'):
                        if temp_depths['size{}'.format(i_size + 1)][ne] != 'None':
                            max_depth = int(temp_depths['size{}'.format(i_size + 1)][ne])
                        else:
                            print('size{} max_depth = None'.format(i_size + 1))
                            max_depth = 1

                    print('\n---{} run {}, size {}, {} max_depth: {}---'.format(subject_system, ne+1, i_size+1, mode, max_depth))

                    if max_depth <= min_depth:
                        print('d={} is samller than the min_depth {}\n'.format(max_depth, min_depth))
                    else:
                        if max_depth == 0:
                            learning_model = 'HINNPerf'
                        elif max_depth >= 5:
                            test_mode = True

                        file_start = '{}_{}_d{}_{}-{}_{}'.format(learning_model, subject_system, max_depth, N_train, N_test,
                                                                 seed)
                        saving_dir = '{}/{}.csv'.format(saving_folder, file_start)
                        if not os.path.exists('{}/{}.csv'.format(saving_folder, file_start)):
                            saving_table = {'Run': [], 'MRE': [], 'Time': [], 'Time_dividing':[], 'Time_training':[], 'Time_predicting':[], 'num_block': [], 'num_layer_pb':[], 'lamda': [], 'gnorm': [], 'lr': [], 'max_epoch': [max_epoch]}
                            for temp_run in range(30):
                                saving_table['Run'].append(temp_run + 1)
                                saving_table['MRE'].append('None')
                                saving_table['Time'].append('None')
                                saving_table['Time_dividing'].append('None')
                                saving_table['Time_training'].append('None')
                                saving_table['Time_predicting'].append('None')
                                saving_table['num_block'].append([])
                                saving_table['num_layer_pb'].append([])
                                saving_table['lamda'].append([])
                                saving_table['gnorm'].append([])
                                saving_table['lr'].append([])
                                if temp_run != 0:
                                    saving_table['OH'].append(' ')
                                    saving_table['max_epoch'].append(' ')

                            if save_results:
                                pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                                print('Creating {}...'.format(saving_dir))
                        elif os.path.exists('{}/{}.csv'.format(saving_folder, file_start)):
                            saving_table = pd.read_csv(saving_dir).to_dict('list')

                        # if 'Time' not in saving_table.keys():
                        #     saving_table['Time'] = []
                        # if len(saving_table['MRE']) < 30:
                        #     for temp in range(len(saving_table['MRE']), 30):
                        #         saving_table['Run'].append(temp)
                        #         saving_table['MRE'].append('None')
                        #     if save_results:
                        #         pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                        #         print('Creating {}...'.format(saving_dir))
                        # elif len(saving_table['Time']) < 30:
                        #     for temp in range(len(saving_table['Time']), 30):
                        #         saving_table['Time'].append('None')
                        #     if save_results:
                        #         pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                        #         print('Creating {}...'.format(saving_dir))

                        finished = False
                        print(type(saving_table['MRE'][ne]))
                        if isinstance(saving_table['MRE'][ne], float) and isinstance(saving_table['Time'][ne], float):
                            if saving_table['MRE'][ne] > 0 and saving_table['Time'][ne] > 0:
                                print('(float) {} Run {} has finished, MRE: {}, time: {}'.format(file_start, ne + 1,
                                                                                                 saving_table['MRE'][ne],
                                                                                                 saving_table['Time'][ne]))
                                finished = True
                        elif isinstance(saving_table['MRE'][ne], str) and isinstance(saving_table['Time'][ne], str):
                            if len(saving_table['MRE'][ne].split('.')) > 1 and len(saving_table['Time'][ne].split('.')) > 1:
                                print('(string) {} Run {} has finished, MRE: {}, time: {}'.format(file_start, ne + 1,
                                                                                                  saving_table['MRE'][ne],
                                                                                                  saving_table['Time'][ne]))
                                finished = True
                        if not finished:
                            random.seed(ne * seed)

                            # Start measure time
                            start_time = time.time()
                            start_time_dividing = time.time()

                            non_zero_indexes = get_non_zero_indexes(whole_data, total_tasks)

                            testing_index = sample(list(non_zero_indexes), N_test)
                            non_zero_indexes = np.setdiff1d(non_zero_indexes, testing_index)
                            training_index = sample(list(non_zero_indexes), N_train)

                            print('Training sample size: {}'.format(N_train))
                            print('Testing sample size: ', N_test)

                            # compute the weights of each feature using Mutual Information, for eliminating insignificant features
                            weights = []
                            feature_weights = mutual_info_regression(whole_data[non_zero_indexes, 0:N_features],
                                                                     whole_data[non_zero_indexes, -1], random_state=0)
                            # print('Computing weights of {} samples'.format(len(non_zero_indexes)))
                            for i in range(N_features):
                                weight = feature_weights[i]
                                # print('Feature {} weight: {}'.format(i, weight))
                                weights.append(weight)

                            # print('\n---DNN_DaL depth {}---'.format(max_depth))
                            # initialize variables
                            max_X = []
                            max_Y = []
                            config = []
                            rel_errors = []
                            X_train = []
                            Y_train = []
                            X_train1 = []
                            Y_train1 = []
                            X_train2 = []
                            Y_train2 = []
                            X_test = []
                            Y_test = []
                            cluster_indexes_all = []

                            # generate clustering labels based on the dividing conditions of DT
                            print('Dividing...')
                            # get the training X and Y for clustering
                            Y = whole_data[non_zero_indexes, -1][:, np.newaxis]
                            X = whole_data[non_zero_indexes, 0:N_features]

                            # build and train a CART to extract the dividing conditions
                            # DT = build_model('DT', test_mode, X, Y)
                            DT = DecisionTreeRegressor(random_state=seed, criterion='squared_error', splitter='best')
                            DT.fit(X, Y)
                            tree_ = DT.tree_  # get the tree structure

                            # recursively divide samples
                            cluster_indexes_all = recursive_dividing(0, 1, tree_, X, non_zero_indexes, max_depth,
                                                                     min_samples, cluster_indexes_all)

                            k = len(cluster_indexes_all)  # the number of divided subsets

                            end_time_dividing = time.time()
                            time_dividing = ((end_time_dividing - start_time_dividing) / 60)
                            print('Dividing time cost (minutes): {:.2f}'.format(time_dividing))

                            lamdas = [0.001, 0.01, 0.1,
                                      1]  # the list of l2 regularization parameters for hyperparameter tuning
                            gnorms = [True, False]  # gnorm parameters for hyperparameter tuning
                            lrs = [0.0001, 0.001, 0.01]  # the list of learning rates for hyperparameter tuning
                            init_config = dict(
                                # input_dim=[data_gen.config_num],
                                num_neuron=[128],
                                num_block=[2, 3, 4],
                                num_layer_pb=[2, 3, 4],
                                lamda=lamdas,
                                linear=[False],
                                gnorm=gnorms,
                                lr=lrs,
                                decay=[None],
                                verbose=[True]
                            )
                            # if there is only one cluster, DaL can not be used
                            if k <= 1:
                                start_time_training = time.time()
                                print(
                                    'Error: samples are less than the minimum number (min_samples={}), training HINNPerf...'.format(
                                        min_samples))
                                rel_error = get_HINNPerf_MRE([whole_data, training_index, testing_index, test_mode, init_config])
                                print('> HINNPerf MRE: {}'.format(rel_error))

                                end_time_training = time.time()
                                time_training = ((end_time_training - start_time_training) / 60)
                                print('Training time cost (minutes): {:.2f}'.format(time_training))

                                # End measuring time
                                end_time = time.time()
                                total_time = ((end_time - start_time) / 60)
                                time_predicting = total_time - time_training - time_dividing
                                print('Predicting time cost (minutes): {:.2f}'.format(time_predicting))
                                print('Total time cost (minutes): {:.2f}'.format(total_time))

                                if save_results:
                                    saving_dir = '{}/{}.csv'.format(saving_folder, file_start)
                                    saving_table = pd.read_csv(saving_dir).to_dict('list')
                                    saving_table['Run'][ne] = ne + 1
                                    saving_table['MRE'][ne] = rel_error
                                    saving_table['Time'][ne] = total_time
                                    saving_table['Time_dividing'][ne] = time_dividing
                                    saving_table['Time_training'][ne] = time_training
                                    saving_table['Time_predicting'][ne] = time_predicting
                                    pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                                    print('Saving to {}...'.format(saving_dir))
                            else:
                                # extract training samples from each cluster
                                N_trains = []  # N_train for each cluster
                                cluster_indexes = []
                                for i in range(k):
                                    if int(N_train) > len(cluster_indexes_all[i]):  # if N_train is too big
                                        N_trains.append(int(len(cluster_indexes_all[i])))
                                    else:
                                        N_trains.append(int(N_train))
                                    # sample N_train samples from the cluster
                                    cluster_indexes.append(random.sample(cluster_indexes_all[i], N_trains[i]))
    
                                # generate the samples and labels for classification
                                total_index = cluster_indexes[0]  # samples in the first cluster
                                clusters = np.zeros(int(len(cluster_indexes[0])))  # labels for the first cluster
                                for i in range(k):
                                    if i > 0:  # the samples and labels for each cluster
                                        total_index = total_index + cluster_indexes[i]
                                        clusters = np.hstack((clusters, np.ones(int(len(cluster_indexes[i]))) * i))
    
                                # get max_X and max_Y for scaling
                                max_X = np.amax(whole_data[total_index, 0:N_features], axis=0)  # scale X to 0-1
                                if 0 in max_X:
                                    max_X[max_X == 0] = 1
                                max_Y = np.max(whole_data[total_index, -1]) / 100  # scale Y to 0-100
                                if max_Y == 0:
                                    max_Y = 1
    
                                # get the training data for each cluster
                                for i in range(k):  # for each cluster
                                    temp_X = whole_data[cluster_indexes[i], 0:N_features]
                                    temp_Y = whole_data[cluster_indexes[i], -1][:, np.newaxis]
                                    # Scale X and Y
                                    X_train.append(np.divide(temp_X, max_X))
                                    Y_train.append(np.divide(temp_Y, max_Y))
                                X_train = np.array(X_train)
                                Y_train = np.array(Y_train)
    
                                # get the testing data
                                X_test = whole_data[testing_index, 0:N_features]
                                X_test = np.divide(X_test, max_X)  # scale X
                                Y_test = whole_data[testing_index, -1][:, np.newaxis]
    
                                # split train data into 2 parts for hyperparameter tuning
                                for i in range(0, k):
                                    N_cross = int(np.ceil(X_train[i].shape[0] * 2 / 3))
                                    X_train1.append(X_train[i][0:N_cross, :])
                                    Y_train1.append(Y_train[i][0:N_cross, :])
                                    X_train2.append(X_train[i][N_cross:N_trains[i], :])
                                    Y_train2.append(Y_train[i][N_cross:N_trains[i], :])
    
                                # process the sample to train a classification model
                                X_smo = whole_data[total_index, 0:N_features]
                                y_smo = clusters
                                for j in range(N_features):
                                    X_smo[:, j] = X_smo[:, j] * weights[j]  # assign the weight for each feature
    
                                # SMOTE is an oversampling algorithm when the sample size is too small
                                enough_data = True
                                for i in range(0, k):
                                    if len(X_train[i]) < 5:
                                        enough_data = False
                                if enough_data:
                                    smo = SMOTE(random_state=1, k_neighbors=3)
                                    X_smo, y_smo = smo.fit_resample(X_smo, y_smo)
    
                                print('Training RF classifier...')
                                # build a random forest classifier to classify testing samples
                                forest = RandomForestClassifier(random_state=seed, criterion='gini')
                                # tune the hyperparameters if not in test mode
                                if (not test_mode) and enough_data:
                                    param = {'n_estimators': np.arange(10, 100, 10)}
                                    gridS = GridSearchCV(forest, param)
                                    gridS.fit(X_smo, y_smo)
                                    print(gridS.best_params_)
                                    forest = RandomForestClassifier(**gridS.best_params_, random_state=seed, criterion='gini')
                                forest.fit(X_smo, y_smo)  # training
    
                                # classify the testing samples
                                testing_clusters = []  # classification labels for the testing samples
                                for i in range(0, k):
                                    testing_clusters.append([])
                                X = whole_data[testing_index, 0:N_features]
                                for j in range(N_features):
                                    X[:, j] = X[:, j] * weights[j]  # assign the weight for each feature
                                print('Predicting testing samples')
                                for temp_index in testing_index:
                                    temp_X = whole_data[temp_index, 0:N_features]
                                    temp_cluster = forest.predict(
                                        temp_X.reshape(1, -1))  # predict the dedicated local DNN using RF
                                    testing_clusters[int(temp_cluster)].append(temp_index)
                                print('Prediction finished')
                                # print('Testing size: ', len(testing_clusters))
                                # print('Testing sample clusters: {}'.format((testing_clusters)))
    
                                ### Train DNN_DaL
                                print('Training Local models...')
                                start_time_training = time.time()
                                ## tune DNN for each cluster (division) with multi-thread
                                from concurrent.futures import ThreadPoolExecutor
    
                                # create a multi-thread pool
                                with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
                                    args = []  # prepare arguments for hyperparameter tuning
                                    for i in range(k):  # for each division
                                        args.append([whole_data, cluster_indexes[i], testing_clusters[i], test_mode, init_config])
                                    # optimal_params contains the results from the function 'hyperparameter_tuning'
                                    for i, best_config in enumerate(pool.map(get_HINNPerf_best_config, args)):
                                        print('Tuning division {}... ({} samples)'.format(i + 1, len(X_train[i])))
                                        config.append(best_config)
    
                                for i in range(k):
                                    print('Learning division {}... ({} samples)'.format(i + 1, len(X_train[i])))
    
                                    # train a local DNN model using the optimal hyperparameters
                                    data_gen = DataPreproc(whole_data, cluster_indexes[i], testing_clusters[i])
                                    runner = ModelRunner(data_gen, MLPHierarchicalModel, max_epoch=max_epoch)
                                    rel_error = runner.get_rel_error(config[i])
                                    rel_errors += list(rel_error)
                                rel_errors = np.mean(rel_errors) * 100
                                # compute the MRE (MAPE) using the testing samples
                                print('Testing...')
                                print('> DaL_HINNPerf MRE: {}'.format(round(rel_errors, 2)))
    
                                end_time_training = time.time()
                                time_training = ((end_time_training - start_time_training) / 60)
                                print('Training time cost (minutes): {:.2f}'.format(time_training))
    
                                # End measuring time
                                end_time = time.time()
                                total_time = ((end_time - start_time) / 60)
                                print('Time cost (minutes): {:.2f}'.format(total_time))
    
                                time_predicting = total_time - time_training - time_dividing
    
                                if save_results:
                                    saving_dir = '{}/{}.csv'.format(saving_folder, file_start)
                                    saving_table = pd.read_csv(saving_dir).to_dict('list')
                                    saving_table['num_block'][ne] = []
                                    saving_table['num_layer_pb'][ne] = []
                                    saving_table['lamda'][ne] = []
                                    saving_table['gnorm'][ne] = []
                                    saving_table['lr'][ne] = []
                                    for i in range(k):
                                        print('Learning division {}... ({} samples)'.format(i + 1, len(X_train[i])))
                                        saving_table['num_block'][ne].append(config[i]['num_block'])
                                        saving_table['num_layer_pb'][ne].append(config[i]['num_layer_pb'])
                                        saving_table['lamda'][ne].append(config[i]['lamda'])
                                        saving_table['gnorm'][ne].append(config[i]['gnorm'])
                                        saving_table['lr'][ne].append(config[i]['lr'])
                                    saving_table['Run'][ne] = ne + 1
                                    saving_table['MRE'][ne] = rel_errors
                                    saving_table['Time'][ne] = total_time
                                    saving_table['Time_dividing'][ne] = time_dividing
                                    saving_table['Time_training'][ne] = time_training
                                    saving_table['Time_predicting'][ne] = time_predicting
                                    pd.DataFrame(saving_table).to_csv(saving_dir, index=False)
                                    print('Saving to {}...'.format(saving_dir))
