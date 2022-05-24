import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

# column_names = ['emg']
# column_names = ['rms', 'mav', 'areg1', 'areg2', 'areg3', 'areg4']
# filename = 'svm_ecr_model3.sav'
# filename = 'svm_ecr_fs.sav'
filename = 'svm_ecr_fsenv.sav'

# def svm_learning():
#
#     data_f1 = pd.read_csv('fcu_relax_opt1.txt', sep=',', names=column_names)
#     data_f2 = pd.read_csv('fcu_w_opt1.txt', sep=',', names=column_names)
#     # data_f1 = pd.read_csv('data14opt.txt', names=column_names, skiprows=50, skipfooter=50, engine='python')
#     # data_f2 = pd.read_csv('data11opt.txt', names=column_names, skiprows=50, skipfooter=50, engine='python')
#
#     n1 = data_f1.shape[0]
#     n2 = data_f2.shape[0]
#     listofzeros = [0] * n1
#     listofones = [1] * n2
#
#     data_f = pd.concat([data_f1, data_f2], ignore_index=True, sort=False)
#     listofzeros.extend(listofones)
#
#     classif = pd.DataFrame(listofzeros, columns=['class'])
#
#     #Separating data to train and test
#     X_train, X_test, y_train, y_test = train_test_split(data_f, classif, test_size=0.33, random_state=0)
#
#     #Scaling and centering the train and test data
#     X_train_scaled = scale(X_train)
#     X_test_scaled = scale(X_test)
#
#     #Building a preliminary SVM
#     clf_svm = SVC(random_state=0)
#     clf_svm.fit(X_train_scaled, y_train.values.ravel())
#     #fitting on the training data
#     ConfusionMatrixDisplay.from_estimator(clf_svm,
#                       X_test_scaled,
#                       y_test,
#                       display_labels=["wrist", "wrist_right"]
#                       )
#     plt.show()

    # #Improving SVM by changing parameters
    # param_grid = [
    #     {'C': [0.5, 1, 10, 100], #regularization param
    #     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
    #     'kernel': ['rbf']},
    # ]
    #
    # optimal_params = GridSearchCV(SVC(),
    #                             param_grid,
    #                             cv=5, #cross validation
    #                             scoring='accuracy', #change to improve SVM
    #                             verbose=0)
    # optimal_params.fit(X_train, y_train.values.ravel())
    # print(optimal_params.best_params_)
    #
    # # clf_svm = SVC(random_state=0, C=10, gamma=0.0001)
    # clf_svm = SVC(random_state=0, C=100, gamma=0.001)
    # clf_svm.fit(X_train, y_train.values.ravel())
    #
    # Saving learned SVM model
    # pickle.dump(clf_svm, open(filename, 'wb'))
    #
    # # Plotting results
    # ConfusionMatrixDisplay.from_estimator(clf_svm,
    #                     X_test_scaled,
    #                     y_test,
    #                     display_labels=["wrist", "wrist_right"]
    #                     )
    #
    # plt.show()

#svm_learning()

def svm_processing(x_test):
    # x_test_scaled = scale(x_test)
    #print(x_test_scaled)
    # Opening saved SVM model
    loaded_model = pickle.load(open(filename, 'rb'))
    # Predicting the results of classification
    # x_test = x_test.reshape(-1, 1)
    y_pred = loaded_model.predict(x_test)
    m = np.bincount(y_pred).argmax()
    # if y_pred[0] + y_pred[1] == 4:
    #     m = 4
    print(y_pred)
    return m
