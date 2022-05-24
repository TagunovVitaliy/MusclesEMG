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
from sklearn.tree import DecisionTreeClassifier
import itertools as it
import pickle
import seaborn as sns

filename = 'svm_ecr_fsenv.sav' # saved model name

def svm_after_fselection(file_names, fs_complete):
    # Preparing collected processed data from files:
    listall = []
    data_f = pd.DataFrame()
    for i in range(0, len(file_names)):
        file_data = pd.read_csv(file_names[i], sep=',', names=fs_complete)
        n = file_data.shape[0]
        listofclass = [i] * n
        listall = list(it.chain(listall, listofclass))
        data_f = pd.concat([data_f, file_data], ignore_index=True, sort=False)
    # print(data_f)
    # Y data for classificator:
    classif = pd.DataFrame(listall, columns=['class']) # y data - class values data frame
    # Splitting X data to test and train:
    X_train, X_test, y_train, y_test = train_test_split(
        data_f, classif, random_state=0)
    # Scaling (normalizing) data:
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    # Creating SVM model:
    clf_svm = SVC(random_state=0)
    clf_svm.fit(X_train_scaled, y_train.values.ravel())
    clf_svm.predict(X_train_scaled)

    y_pred = clf_svm.predict(X_test_scaled)

    # Saving learned SVM model
    # pickle.dump(clf_svm, open(filename, 'wb'))

    # Plotting results of classification:
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                        index=['Class0', 'Class1', 'Class2', 'Class3'],
                        columns=['Class0', 'Class1', 'Class2', 'Class3'])

    print(cm_df)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    # Calculating the accuracy of prediction
    con_mtx = cm_df.to_numpy()
    all, pos = 0, 0
    for i in range(len(con_mtx)):
        for j in range(len(con_mtx[i])):
            all += con_mtx[i][j]
            if i == j:
                pos += con_mtx[i][j]
    acc = pos / all * 100
    print("Accuracy of prediction: " + str(acc) + "%")

    # plt.show()