import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Lab3Functions as l3f
import EMGfunctions as emgf
import SVM as svm
import time as t

column_names = ['emg', 't']
# data_w1 = pd.read_csv('fcu/fcu_l_all.txt', sep=',', names=column_names, skiprows=50, skipfooter=50, engine='python')

def collect_data(file_name): # for data collection and learning
    data_w1 = pd.read_csv(file_name, sep=',', names=column_names, skiprows=50, skipfooter=50, engine='python')
    emg = data_w1.emg
    time = data_w1.t
    emg_correctmean = emgf.remove_mean(emg, time)
    # emg_filtered = emgf.emg_filter(emg_correctmean, time)
    # emg_rectified = emgf.emg_rectify(emg_filtered, time)

    emg_filtered, emg_rectified, emg_envelope = emgf.alltogether(time, emg_correctmean)

    # emg_rms, emg_wl, areg1, areg2, areg3, areg4, iemg, mav2, ssi, var = emgf.splitter(emg_envelope, emg_filtered)
    # emg_rms, emg_wl, areg1, areg2, areg3, areg4, iemg, mav2, ssi, var = emgf.window(emg_rectified)
    # emg_features = (np.vstack((emg_rms, emg_wl, areg1, areg2, areg3, areg4, iemg, mav2, ssi, var))).T
    emg_features = emgf.window(emg_filtered)

    np.savetxt('fcu/fcu_l_opt1win.txt', emg_features, fmt='%.5f', delimiter=',')
    plt.show()

def data_processing(emg_dataset): # for real time
    data_w1 = pd.DataFrame(emg_dataset, columns=['emg', 't'])
    emg = data_w1.emg
    time = data_w1.t
    emg_correctmean = emgf.remove_mean(emg, time)
    emg_envelope = emgf.alltogether(time, emg_correctmean)

    # emg_rms, emg_wl, areg1, areg2, areg3, areg4 = emgf.splitter1(emg_rectified)
    # emg_features = (np.vstack((emg_rms, emg_wl, areg1, areg2, areg3, areg4))).T
    # emg_rms, emg_wl, areg1, areg3, iemg, ssi, var = emgf.splitter2(emg_rectified)
    # emg_features = (np.vstack((emg_rms, emg_wl, areg1, areg3, iemg, ssi, var))).T

    emg_rms, iemg, mav2, ssi, var = emgf.splitter3(emg_envelope)
    emg_features = (np.vstack((emg_rms, iemg, mav2, ssi, var))).T

    x = svm.svm_processing(emg_features)
    return x