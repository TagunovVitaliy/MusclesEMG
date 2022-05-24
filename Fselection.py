from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import itertools as it
from feature_engine.selection import SelectByShuffling
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import SCORERS
import warnings
import os
import SVMmulti as md
# Comment this to see warning:
warnings.filterwarnings('ignore')
# Names of features from file:
column_names = ['rms', 'wl', 'areg1', 'areg2', 'areg3', 'areg4', 'iemg', 'mav2',
                'ssi', 'var', 'stdev', 'nop', 'mnop', 'dumv', 'mfl', 'per',
                'skew', 'kurt', 'zc', 'ssc', 'wa']
# Different lists of scoring parameters names:
scoring_names = ['balanced_accuracy',
                 'f1_micro', 'precision_micro', 'recall_micro']
scoring_names1 = ['f1_macro', 'precision_macro', 'recall_macro']
scoring_names2 = ['f1_weighted', 'precision_weighted', 'recall_weighted']
scoring_names_other = ['jaccard_micro', 'jaccard_macro', 'jaccard_weighted']
scoring_names_all = ['accuracy', 'balanced_accuracy', 'f1_micro', 'precision_micro', 'recall_micro',
                     'f1_macro', 'precision_macro', 'recall_macro', 'f1_weighted', 'precision_weighted',
                     'recall_weighted', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted']
# scoring_names_all = ['f1_micro', 'recall_micro',
#                      'f1_macro', 'precision_macro', 'recall_macro', 'precision_weighted', 'jaccard_micro']
# File names with previously collected data:
file_names = ['ecr_relax_opt2win.txt', 'ecr_b_opt2win.txt',
             'ecr_f_opt2win.txt', 'ecr_l_opt2win.txt']
# file_names = ['fcu/fcu_relax_opt1win.txt', 'fcu/fcu_b_opt1win.txt',
#              'fcu/fcu_f_opt1win.txt', 'fcu/fcu_l_opt1win.txt']

def feature_sel():
    # Preparing collected processed data from files:
    listall = []
    data_f = pd.DataFrame()
    for i in range(0, len(file_names)):
        file_data = pd.read_csv(file_names[i], sep=',', names=column_names)
        n = file_data.shape[0]
        listofclass = [i] * n
        listall = list(it.chain(listall, listofclass))
        data_f = pd.concat([data_f, file_data], ignore_index=True, sort=False)
    # X and y data for classificator:
    classif = pd.DataFrame(listall, columns=['class']) # y data - class values data frame
    data_f_scaled = scale(data_f) # scaled X data - features data

    # Initializing the SVM model:
    clf_svm = SVC(random_state=0)

    # random_shuff_for_each(clf_svm, data_f_scaled, classif)

    # Feature Selection algorithms:
    random_shuffling(clf_svm, data_f_scaled, classif) # Random shuffling method
    # single_feature(clf_svm, data_f_scaled, classif) # Select by single feature
    # wrapper(clf_svm, data_f_scaled, classif) # Wrapper method

def random_shuff_for_each(clf_svm, data_f, classif):
    for i in range(0, len(scoring_names_all)):
        tr = SelectByShuffling(estimator=clf_svm, scoring=scoring_names_all[i], cv=3)
        tr.fit(data_f, classif.values.ravel())
        features_to_drop = tr.features_to_drop_
        columns_drop1 = []
        complete_col1 = ['rms', 'wl', 'areg1', 'areg2', 'areg3', 'areg4', 'iemg', 'mav2',
                'ssi', 'var', 'stdev', 'nop', 'mnop', 'dumv', 'mfl', 'per',
                'skew', 'kurt', 'zc', 'ssc', 'wa']
        for j in range(0, len(features_to_drop)):
            s = int(features_to_drop[j])
            columns_drop1.append(column_names[s])
            complete_col1.remove(column_names[s])
        print("The results for: " + scoring_names_all[i])
        print("Columns to drop:")
        print(columns_drop1)
        drop_features(columns_drop1, complete_col1)

def random_shuffling(clf_svm, data_f, classif):
    print("Select by random shuffling method:\n")
    tr1 = SelectByShuffling(estimator=clf_svm, scoring='accuracy', cv=3) # Printing scores of "accuracy" just for comparison
    tr1.fit(data_f, classif.values.ravel())
    print("Scoring results for accuracy:")
    print(tr1.performance_drifts_)
    features_to_drop = tr1.features_to_drop_
    print("Numbers of bad features to drop:")
    print(features_to_drop)
    # columns_drop = ['rms', 'areg1', 'areg2', 'areg4', 'mav2', 'ssi', 'mfl', 'per', 'skew', 'kurt', 'zc', 'ssc']
    # complete_col = ['wl', 'areg3', 'iemg', 'var', 'stdev', 'nop', 'mnop', 'dumv', 'wa']
    # drop_features(columns_drop, complete_col)
    # Для сравнения с остальными:
    columns_drop1 = []
    # complete_col1 = ['rms', 'wl', 'areg1', 'areg2', 'areg3', 'areg4', 'iemg', 'mav2',
    #             'ssi', 'var', 'stdev', 'nop', 'mnop', 'dumv', 'mfl', 'per',
    #             'skew', 'kurt', 'zc', 'ssc', 'wa']
    for i in range(0, len(features_to_drop)):
        s = int(features_to_drop[i])
        columns_drop1.append(column_names[s])
        # complete_col1.remove(column_names[s])
    print(columns_drop1)
    # print(complete_col1)
    # drop_features(columns_drop1, complete_col1)

    # Calculating random shuffling for all scoring parameters:
    data_all = pd.DataFrame()
    for i in range(0, len(scoring_names_all)):
        tr = SelectByShuffling(estimator=clf_svm, scoring=scoring_names_all[i], cv=3)
        tr.fit(data_f, classif.values.ravel())
        data_all[scoring_names_all[i]] = tr.performance_drifts_.values()
    print("Scoring results for each parameter:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data_all)

    # Using ML clustering algorithm to separate result data:
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data_all)
    labels_to_drop = kmeans.labels_
    # print(labels_to_drop)
    # Checking the centers of clustering:
    if kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0]:
        for i in range(0, len(labels_to_drop)):
            if labels_to_drop[i] == 0:
                labels_to_drop[i] = 1
            else:
                labels_to_drop[i] = 0

    columns_drop, complete_col = [], []
    for i in range(0, len(labels_to_drop)):
        if labels_to_drop[i] == 0:
            columns_drop.append(column_names[i])
        elif labels_to_drop[i] == 1:
            complete_col.append(column_names[i])
    print("Features to drop after clustering:")
    print(columns_drop)
    print("Features to not drop after clustering:")
    print(complete_col)
    # columns_drop = ['areg1', 'areg4', 'mav2', 'mnop', 'mfl', 'kurt', 'zc', 'ssc']
    # complete_col = ['rms', 'wl', 'areg2', 'areg3', 'iemg', 'ssi', 'var', 'stdev', 'nop', 'dumv', 'per', 'skew', 'wa']
    drop_features(columns_drop, complete_col)

    # data_fs1 = data_f1.drop(labels=columns_drop, axis=1).to_numpy()
    # data_fs2 = data_f2.drop(labels=columns_drop, axis=1).to_numpy()
    # data_fs3 = data_f3.drop(labels=columns_drop, axis=1).to_numpy()
    # data_fs4 = data_f4.drop(labels=columns_drop, axis=1).to_numpy()
    # data_fs5 = data_f5.drop(labels=columns_drop, axis=1).to_numpy()
    # data_fs6 = data_f6.drop(labels=columns_drop, axis=1).to_numpy()

    # np.savetxt('ecr_relax_opt5fs.txt', data_fs1, fmt='%.5f', delimiter=',')
    # np.savetxt('ecr_b_opt5fs.txt', data_fs3, fmt='%.5f', delimiter=',')
    # np.savetxt('ecr_f_opt5fs.txt', data_fs4, fmt='%.5f', delimiter=',')
    # np.savetxt('ecr_l_opt5fs.txt', data_fs5, fmt='%.5f', delimiter=',')
    #
    # np.savetxt('ecr_r_opt5fs.txt', data_fs6, fmt='%.5f', delimiter=',')
    # np.savetxt('ecr_w_opt5fs.txt', data_fs2, fmt='%.5f', delimiter=',')

def single_feature(clf_svm, data_f, classif):
    print("--------------------------------------------------------------------------------------------\n")
    print("Select by single feature method:\n")
    data_all1 = pd.DataFrame()
    for i in range(0, len(scoring_names_all)):
        sel = SelectBySingleFeaturePerformance(estimator=clf_svm, scoring=scoring_names_all[i], cv=3)
        sel.fit(data_f, classif.values.ravel())
        print(sel.features_to_drop_)
        data_all1[scoring_names_all[i]] = sel.feature_performance_.values()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data_all1)

    # Using ML clustering algorithm to separate result data
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data_all1)
    labels_to_drop = kmeans.labels_
    print(labels_to_drop)
    # Checking the centers of clustering:
    if kmeans.cluster_centers_[0, 0] > kmeans.cluster_centers_[1, 0]:
        for i in range(0, len(labels_to_drop)):
            if labels_to_drop[i] == 0:
                labels_to_drop[i] = 1
            else:
                labels_to_drop[i] = 0
    # Searching what columns we need to drop
    columns_drop, complete_col = [], []
    for i in range(0, len(labels_to_drop)):
        if labels_to_drop[i] == 0:
            columns_drop.append(column_names[i])
        elif labels_to_drop[i] == 1:
            complete_col.append(column_names[i])
    print("Features to drop after clustering:")
    print(columns_drop)
    print("Features to not drop after clustering:")
    print(complete_col)
    # columns_drop = ['wl', 'areg1', 'areg2', 'areg3', 'areg4', 'mav2', 'ssi', 'var']
    # complete_col = ['rms', 'iemg']
    drop_features(columns_drop, complete_col)

def wrapper(clf_svm, data_f, classif):
    print("--------------------------------------------------------------------------------------------\n")
    print("Wrapper selection method:\n")
    data_all2 = np.empty([0])
    all_feat = np.empty([0])
    data_all = {}
    for i in range(0, len(scoring_names_all)): #scoring_names_all
        sfs1 = SFS(clf_svm,
                   k_features=1,
                   forward=False,
                   verbose=2,
                   scoring=scoring_names_all[i],
                   cv=3)
        sfs1.fit(data_f, classif.values.ravel())
        nested_dict = sfs1.subsets_
        print(nested_dict) # Results for each scoring parameter
        max = 0.0
        indx = 0
        for j in range(1, len(nested_dict)):
            if nested_dict[j]['avg_score'] > max:
                max = nested_dict[j]['avg_score']
                indx = j
        best_feats = nested_dict[indx]['feature_idx']
        data_all2 = np.append(data_all2, max)
        all_feat = np.append(all_feat, best_feats)
    print("The best feature numbers chosen by the wrapper:")
    print(all_feat)
    # The array with all feature numbers:
    # for i in range(0, len(column_names)):
    #     all_num = np.append(all_num, i)
    sum = 0
    for i in range(0, len(scoring_names_all)):
        for j in range(0, len(all_feat)):
            if i == int(all_feat[j]):
                sum += 1
        data_all[column_names[i]] = sum
        sum = 0
    print(data_all)
    # Using ML clustering algorithm to separate result data
    # kmeans = KMeans(n_clusters=2, random_state=0)
    # kmeans.fit(data_all)
    # labels_to_drop = kmeans.labels_
    # print(labels_to_drop)

def drop_features(fs_to_drop, fs_complete):
    save_path = 'fs_data/'
    file_name = 'fs_'
    fs_file_names = []
    for i in range(0, len(file_names)):
        file_data = pd.read_csv(file_names[i], sep=',', names=column_names)
        file_data = file_data.drop(labels=fs_to_drop, axis=1).to_numpy()
        completeName = os.path.join(save_path, file_name + str(i) + ".txt")
        np.savetxt(completeName, file_data, fmt='%.5f', delimiter=',')
        fs_file_names = np.append(fs_file_names, completeName)
    md.svm_after_fselection(fs_file_names, fs_complete)
    # Deleting files:
    for i in range(0, len(fs_file_names)):
        if os.path.exists(fs_file_names[i]):
            os.remove(fs_file_names[i])
        else:
            print("The file " + fs_file_names[i] + " does not exist")