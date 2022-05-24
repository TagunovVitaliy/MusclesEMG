import matplotlib.pyplot as plt
import numpy as np
import math
import scipy as sp
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg

def remove_mean(emg, time):
    # process EMG signal: remove mean
    emg_correctmean = emg - np.mean(emg)

    # plot comparison of EMG with offset vs mean-corrected values
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Mean offset present')
    plt.plot(time, emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
   #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Mean-corrected values')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    fig.tight_layout()
    fig_name = 'fig2.png'
    fig.set_size_inches(w=11, h=7)
    fig.savefig(fig_name)

    return emg_correctmean

def emg_filter(emg_correctmean, time):
    # create bandpass filter for EMG
    high = 20 / (1000 / 2)
    low = 450 / (1000 / 2)
    b, a = sp.signal.butter(4, [high, low], btype='bandpass')

    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b, a, emg_correctmean)

    # plot comparison of unfiltered vs filtered mean-corrected EMG
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unfiltered EMG')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Filtered EMG')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    fig.tight_layout()
    fig_name = 'fig3.png'
    fig.set_size_inches(w=11, h=7)
    fig.savefig(fig_name)

    return emg_filtered

def emg_rectify(emg_filtered, time):
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # plot comparison of unrectified vs rectified EMG
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unrectified EMG')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Rectified EMG')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    fig.tight_layout()
    fig_name = 'fig4.png'
    fig.set_size_inches(w=11, h=7)
    fig.savefig(fig_name)


def alltogether(time, emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """

    # normalise cut-off frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band, low_band], btype='bandpass')

    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)

    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass / (sfreq / 2)
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

    # plot graphs
    # fig = plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.subplot(1, 3, 1).set_title('Unfiltered,' + '\n' + 'unrectified EMG')
    # plt.plot(time, emg)
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # #plt.ylim(-1.5, 1.5)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('EMG (a.u.)')
    #
    # plt.subplot(1, 3, 2)
    # plt.subplot(1, 3, 2).set_title(
    #     'Filtered,' + '\n' + 'rectified EMG: ' + str(int(high_band * sfreq)) + '-' + str(int(low_band * sfreq)) + 'Hz')
    # plt.plot(time, emg_rectified)
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # #plt.ylim(-1.5, 1.5)
    # #plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
    # plt.xlabel('Time (sec)')
    #
    # plt.subplot(1, 3, 3)
    # plt.subplot(1, 3, 3).set_title(
    #     'Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(int(low_pass * sfreq)) + ' Hz')
    # plt.plot(time, emg_envelope)
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # #plt.ylim(-1.5, 1.5)
    # #plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
    # plt.xlabel('Time (sec)')

    # plt.subplot(1, 4, 4)
    # plt.subplot(1, 4, 4).set_title('Focussed region')
    # plt.plot(time[int(0.9 * 1000):int(1.0 * 1000)], emg_envelope[int(0.9 * 1000):int(1.0 * 1000)])
    # plt.locator_params(axis='x', nbins=4)
    # plt.locator_params(axis='y', nbins=4)
    # plt.xlim(0.9, 1.0)
    #plt.ylim(-1.5, 1.5)
    #plt.xlabel('Time (sec)')

    # fig_name = 'fig_' + str(int(low_pass * sfreq)) + '.png'
    # fig.set_size_inches(w=11, h=7)
    # fig.savefig(fig_name)
    emg_filtered, emg_rectified,
    return emg_envelope

def integratedEMG(array):
    n = array.size
    iemg = 0
    for i in range(0, n):
        iemg += abs(array[i])
    return iemg

def rmsValue(array):
    n = array.size
    squre = 0.0
    root = 0.0
    mean = 0.0

    # calculating Square
    for i in range(0, n):
        squre += (array[i] ** 2)
    # Calculating Mean
    mean = (squre / (float)(n))
    # Calculating Root
    root = math.sqrt(mean)
    return root

def wavLen(array): #Waveform length
    n = array.size
    wl = 0
    for i in range(0, n-1):
        wl += abs(array[i+1]-array[i])
    return wl

def aregression(array):
    # train autoregression
    model = AutoReg(array, lags=3)
    model_fit = model.fit()
    #print('Coefficients: %s' % model_fit.params)
    return model_fit.params

def mavValue2(array):
    n = array.size
    mmav2 = 0.0
    for i in range(0, n):
        if (0.25*n > i):
            w = 4*i/n
        elif (0.75*n < i):
            w = 4*(i-n)/n
        else:
            w = 1
        mmav2 += w * abs(array[i])
    mmav2 = mmav2/n
    return mmav2

def ssIntegral(array): #Simple Square Integral
    n = array.size
    ssi = 0.0
    for i in range(0, n):
        ssi += abs(array[i]) ** 2
    return ssi

def variance(array):
    n = array.size
    var = 0.0
    for i in range(0, n):
        var += array[i] ** 2
    var = var/(n-1)
    return var

def standardDev(array):
    std = np.std(array)
    return std

def numofPeaks(array):
    n = array.size
    nop = 0
    mean_list = []
    rms = rmsValue(array)
    for i in range(0, n):
        if array[i] > rms:
            mean_list = np.append(mean_list, array[i])
            nop += 1
    mean = np.mean(mean_list)
    return nop, mean

def dumvValue(array): #Difference absolute mean value
    n = array.size
    dumv = 0.0
    for i in range(0, n-1):
        dumv += abs(array[i+1] - array[i])
    dumv = dumv/n
    return dumv

def mflValue(array): #Maximum fractal length
    n = array.size
    mfl = 0.0
    for i in range(0, n-1):
        mfl += (array[i+1] - array[i]) ** 2
    mfl = math.log10(math.sqrt(mfl))
    return mfl

def percent(array): #Percentile (Perc)
    perc = np.percentile(array, 75)
    return perc

def mValue(array, k): #Calculates M value for skew and kurt
    n = array.size
    m = 0.0
    mean = np.mean(array)
    for i in range(0, n):
        m += (array[i] - mean) ** k
    m = m/n
    return m

def skewness(array): #Skewness (Skew)
    m2 = mValue(array, 2)
    m3 = mValue(array, 3)
    skew = m3/(m2 * math.sqrt(m2))
    return skew

def kurtosis(array): #Kurtosis (Kurt)
    m2 = mValue(array, 2)
    m4 = mValue(array, 4)
    kurt = m4/(m2*m2)
    return kurt

def zeroCrossing(array): #Zero crossing (ZC)
    n = array.size
    zc = 0
    for i in range(1, n):
        if -(array[i]) * array[i-1] > 0:
            zc += 1
    return zc

def sscValue(array): #Slope sign changes (SSC)
    n = array.size
    ssc = 0
    tr = 100 # threshold
    for i in range(1, n-1):
        f = (array[i] - array[i-1]) * (array[i] - array[i+1])
        if f > tr:
            ssc += 1
    return ssc

def wilsonAmplitude(array):
    n = array.size
    wa = 0
    tr = 100
    for i in range(0, n-1):
        f = abs(array[i] - array[i+1])
        if f > tr:
            wa += 1
    return wa


def splitter(emg, emgf):
    rmsValueAll = wlValueAll = aregValueAll1 = aregValueAll2 = \
        aregValueAll3 = aregValueAll4 = iemgValueAll = mav2ValueAll = \
        ssiValueAll = varValueAll = zcValueAll = waValueAll = np.empty([0])
    for i in range(0, 100 * (emg.size // 100), 100):
        new_emg = emg[i:i+100]
        rmsValueAll = np.append(rmsValueAll, rmsValue(new_emg))
        wlValueAll = np.append(wlValueAll, wavLen(new_emg))
        coefArr = aregression(new_emg)
        aregValueAll1 = np.append(aregValueAll1, coefArr[0])
        aregValueAll2 = np.append(aregValueAll2, coefArr[1])
        aregValueAll3 = np.append(aregValueAll3, coefArr[2])
        aregValueAll4 = np.append(aregValueAll4, coefArr[3])
        iemgValueAll = np.append(iemgValueAll, integratedEMG(new_emg))
        mav2ValueAll = np.append(mav2ValueAll, mavValue2(new_emg))
        ssiValueAll = np.append(ssiValueAll, ssIntegral(new_emg))
        varValueAll = np.append(varValueAll, variance(new_emg))
        # zcValueAll = np.append(zcValueAll, zeroCrossing(emgf))
        # waValueAll = np.append(waValueAll, wilsonAmplitude(emgf))
    return rmsValueAll, wlValueAll, aregValueAll1, aregValueAll2, \
           aregValueAll3, aregValueAll4, iemgValueAll, mav2ValueAll, \
           ssiValueAll, varValueAll

def window(emg):
    rmsValueAll, wlValueAll, aregValueAll1, aregValueAll2, \
        aregValueAll3, aregValueAll4, iemgValueAll, mav2ValueAll, \
        ssiValueAll, varValueAll, stdevValueAll, nopValueAll, mnopValueAll, \
        dumvValueAll, mflValueAll, percentValueAll, skewValueAll,\
        kurtValueAll, zcValueAll, sscValueAll, waValueAll = (np.empty([0]),)*21
    win = 100 # the size of a single window
    shift = 50 # the step of shift
    for i in range(0, emg.size // shift - 1):
        new_emg = emg[i*shift:i*shift+win]
        # new_emg_filt = emg_filtred[i*shift:i*shift+win]
        rmsValueAll = np.append(rmsValueAll, rmsValue(new_emg))
        wlValueAll = np.append(wlValueAll, wavLen(new_emg))
        coefArr = aregression(new_emg)
        aregValueAll1 = np.append(aregValueAll1, coefArr[0])
        aregValueAll2 = np.append(aregValueAll2, coefArr[1])
        aregValueAll3 = np.append(aregValueAll3, coefArr[2])
        aregValueAll4 = np.append(aregValueAll4, coefArr[3])
        iemgValueAll = np.append(iemgValueAll, integratedEMG(new_emg))
        mav2ValueAll = np.append(mav2ValueAll, mavValue2(new_emg))
        ssiValueAll = np.append(ssiValueAll, ssIntegral(new_emg))
        varValueAll = np.append(varValueAll, variance(new_emg))
        stdevValueAll = np.append(stdevValueAll, standardDev(new_emg))
        nopValueAll1, mnopValueAll1 = numofPeaks(new_emg)
        nopValueAll = np.append(nopValueAll, nopValueAll1)
        mnopValueAll = np.append(mnopValueAll, mnopValueAll1)
        dumvValueAll = np.append(dumvValueAll, dumvValue(new_emg))
        mflValueAll = np.append(mflValueAll, mflValue(new_emg))
        percentValueAll = np.append(percentValueAll, percent(new_emg))
        skewValueAll = np.append(skewValueAll, skewness(new_emg))
        kurtValueAll = np.append(kurtValueAll, kurtosis(new_emg))
        zcValueAll = np.append(zcValueAll, zeroCrossing(new_emg))
        sscValueAll = np.append(sscValueAll, sscValue(new_emg))
        waValueAll = np.append(waValueAll, wilsonAmplitude(new_emg))

    emg_features = (np.vstack((rmsValueAll, wlValueAll, aregValueAll1, aregValueAll2,
                               aregValueAll3, aregValueAll4, iemgValueAll, mav2ValueAll,
                               ssiValueAll, varValueAll, stdevValueAll, nopValueAll, mnopValueAll,
                               dumvValueAll, mflValueAll, percentValueAll, skewValueAll,
                               kurtValueAll, zcValueAll, sscValueAll, waValueAll))).T
    return emg_features

def dif_slopes(emg):
    zcValueAll, sscValueAll, waValueAll = (np.empty([0]),)*3
    win = 100 # the size of a single window
    shift = 50 # the step of shift
    for i in range(0, emg.size // shift - 1):
        new_emg = emg[i*shift:i*shift+win]
        zcValueAll = np.append(zcValueAll, zeroCrossing(new_emg))
        sscValueAll = np.append(sscValueAll, sscValue(new_emg))
        waValueAll = np.append(waValueAll, wilsonAmplitude(new_emg))

    return zcValueAll, sscValueAll, waValueAll
