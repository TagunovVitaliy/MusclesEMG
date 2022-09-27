from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import itertools as it
from feature_engine.selection import SelectByShuffling
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import warnings
import os
from SVMlearn import SVMmodel
# Comment this to see warnings:
warnings.filterwarnings('ignore')

class Features:
    # Names of features from file:
    column_names = ['rms', 'wl', 'areg1', 'areg2', 'areg3', 'areg4', 'iemg', 'mav2',
                    'ssi', 'var', 'stdev', 'nop', 'mnop', 'dumv', 'mfl', 'per',
                    'skew', 'kurt', 'zc', 'ssc', 'wa']
    # List of scoring parameters names:
    scoring_names_all = ['accuracy', 'balanced_accuracy', 'f1_micro', 'precision_micro', 'recall_micro',
                         'f1_macro', 'precision_macro', 'recall_macro', 'f1_weighted', 'precision_weighted',
                         'recall_weighted', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted']
    # File names with previously collected data:
    file_names = ['processed/new_opt1.txt', 'processed/new_opt2.txt',
                  'processed/new_opt3.txt', 'processed/new_opt4.txt']

    def feature_sel(self):
        # Preparing collected processed data from files:
        listall = []
        data_f = pd.DataFrame()
        for i in range(0, len(self.file_names)):
            file_data = pd.read_csv(self.file_names[i], sep=',', names=self.column_names)
            n = file_data.shape[0]
            listofclass = [i] * n
            listall = list(it.chain(listall, listofclass))
            data_f = pd.concat([data_f, file_data], ignore_index=True, sort=False)
        # X and y data for classificator:
        classif = pd.DataFrame(listall, columns=['class'])  # y data - class values data frame
        data_f_scaled = scale(data_f)  # scaled X data - features data

        # Initializing the SVM model:
        clf_svm = SVC(random_state=0)

        # Feature Selection algorithms:
        self.random_shuffling(clf_svm, data_f_scaled, classif)  # Random shuffling method
        # single_feature(clf_svm, data_f_scaled, classif) # Select by single feature
        # wrapper(clf_svm, data_f_scaled, classif) # Wrapper method

    def random_shuffling(self, clf_svm, data_f, classif):
        print("Select by random shuffling method:\n")
        tr1 = SelectByShuffling(estimator=clf_svm, scoring='accuracy',
                                cv=3)  # Printing scores of "accuracy" just for comparison
        tr1.fit(data_f, classif.values.ravel())
        print("Scoring results for accuracy:")
        print(tr1.performance_drifts_)
        features_to_drop = tr1.features_to_drop_
        print("Numbers of bad features to drop:")
        print(features_to_drop)

        # Calculating random shuffling for all scoring parameters:
        data_all = pd.DataFrame()
        for i in range(0, len(self.scoring_names_all)):
            tr = SelectByShuffling(estimator=clf_svm, scoring=self.scoring_names_all[i], cv=3)
            tr.fit(data_f, classif.values.ravel())
            data_all[self.scoring_names_all[i]] = tr.performance_drifts_.values()
        print("Scoring results for each parameter:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(data_all)

        # Using ML clustering algorithm to separate result data:
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(data_all)
        labels_to_drop = kmeans.labels_
        # Checking the centers of clustering:
        if kmeans.cluster_centers_[0, 0] > kmeans.cluster_centers_[1, 0]:
            for i in range(0, len(labels_to_drop)):
                if labels_to_drop[i] == 0:
                    labels_to_drop[i] = 1
                else:
                    labels_to_drop[i] = 0

        columns_drop, complete_col = [], []
        for i in range(0, len(labels_to_drop)):
            if labels_to_drop[i] == 0:
                columns_drop.append(self.column_names[i])
            elif labels_to_drop[i] == 1:
                complete_col.append(self.column_names[i])
        print("Features to drop after clustering:")
        print(columns_drop)
        print("Features to not drop after clustering:")
        print(complete_col)

        self.drop_features(columns_drop, complete_col)

    def drop_features(self, fs_to_drop, fs_complete):
        save_path = 'fs_data/'
        file_name = 'fs_'
        fs_file_names = []
        for i in range(0, len(self.file_names)):
            file_data = pd.read_csv(self.file_names[i], sep=',', names=self.column_names)
            file_data = file_data.drop(labels=fs_to_drop, axis=1).to_numpy()
            completeName = os.path.join(save_path, file_name + str(i) + ".txt")
            np.savetxt(completeName, file_data, fmt='%.5f', delimiter=',')
            fs_file_names = np.append(fs_file_names, completeName)
        SVMmodel().svm_after_fselection(fs_file_names, fs_complete)
        # Deleting files:
        for i in range(0, len(fs_file_names)):
            if os.path.exists(fs_file_names[i]):
                os.remove(fs_file_names[i])
            else:
                print("The file " + fs_file_names[i] + " does not exist")
        for i in self.file_names:
            if os.path.exists(i):
                os.remove(i)
            else:
                print("The file " + i + " does not exist")

    def single_feature(self, clf_svm, data_f, classif):
        print("--------------------------------------------------------------------------------------------\n")
        print("Select by single feature method:\n")
        data_all1 = pd.DataFrame()
        for i in range(0, len(self.scoring_names_all)):
            sel = SelectBySingleFeaturePerformance(estimator=clf_svm, scoring=self.scoring_names_all[i], cv=3)
            sel.fit(data_f, classif.values.ravel())
            print(sel.features_to_drop_)
            data_all1[self.scoring_names_all[i]] = sel.feature_performance_.values()
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
                columns_drop.append(self.column_names[i])
            elif labels_to_drop[i] == 1:
                complete_col.append(self.column_names[i])
        print("Features to drop after clustering:")
        print(columns_drop)
        print("Features to not drop after clustering:")
        print(complete_col)
        self.drop_features(columns_drop, complete_col)

    def wrapper(self, clf_svm, data_f, classif):
        print("--------------------------------------------------------------------------------------------\n")
        print("Wrapper selection method:\n")
        data_all2 = np.empty([0])
        all_feat = np.empty([0])
        data_all = {}
        for i in range(0, len(self.scoring_names_all)):  # scoring_names_all
            sfs1 = SFS(clf_svm,
                       k_features=1,
                       forward=False,
                       verbose=2,
                       scoring=self.scoring_names_all[i],
                       cv=3)
            sfs1.fit(data_f, classif.values.ravel())
            nested_dict = sfs1.subsets_
            print(nested_dict)  # Results for each scoring parameter
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
        # Array with all feature numbers:
        sum = 0
        for i in range(0, len(self.scoring_names_all)):
            for j in range(0, len(all_feat)):
                if i == int(all_feat[j]):
                    sum += 1
            data_all[self.column_names[i]] = sum
            sum = 0
        print(data_all)