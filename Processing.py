import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy as sp
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg
from SVMlearn import SVMmodel

class Processing:
    column_names = ['emg', 't']

    def collect_data(self, file_names):
        opt_file_names = []
        for file_name in file_names:
            data_w1 = pd.read_csv(file_name, sep=',', names=self.column_names, skiprows=50, skipfooter=50, engine='python')
            emg = data_w1.emg
            time = data_w1.t

            emg_correctmean = self.remove_mean(emg)
            emg_filtered, emg_rectified, emg_envelope = self.alltogether(time, emg_correctmean)
            emg_features = self.window(emg_filtered)

            new_opt_file = 'processed/new_opt' + file_name[17] + '.txt'
            opt_file_names.append(new_opt_file)
            np.savetxt(new_opt_file, emg_features, fmt='%.5f', delimiter=',')
            plt.show()

    def process_data(self, emg_dataset):
        data_w1 = pd.DataFrame(emg_dataset, columns=['emg', 't'])
        emg = data_w1.emg
        time = data_w1.t
        emg_correctmean = self.remove_mean(emg)
        emg_envelope, emg_filtered, emg_rectified = self.alltogether(time, emg_correctmean)
        emg_features = self.window(emg_filtered)

        x = SVMmodel().svm_processing(emg_features)
        return x

    def remove_mean(self, emg):
        # process EMG signal: remove mean
        emg_correctmean = emg - np.mean(emg)
        return emg_correctmean

    def emg_filter(self, emg_correctmean, time):
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
        # plt.ylim(-1.5, 1.5)
        plt.xlabel('Time (sec)')
        plt.ylabel('EMG (a.u.)')

        plt.subplot(1, 2, 2)
        plt.subplot(1, 2, 2).set_title('Filtered EMG')
        plt.plot(time, emg_filtered)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        # plt.ylim(-1.5, 1.5)
        plt.xlabel('Time (sec)')
        plt.ylabel('EMG (a.u.)')

        fig.tight_layout()
        fig_name = 'fig3.png'
        fig.set_size_inches(w=11, h=7)
        fig.savefig(fig_name)

        return emg_filtered

    def emg_rectify(self, emg_filtered, time):
        # process EMG signal: rectify
        emg_rectified = abs(emg_filtered)

        # plot comparison of unrectified vs rectified EMG
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 1).set_title('Unrectified EMG')
        plt.plot(time, emg_filtered)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        # plt.ylim(-1.5, 1.5)
        plt.xlabel('Time (sec)')
        plt.ylabel('EMG (a.u.)')

        plt.subplot(1, 2, 2)
        plt.subplot(1, 2, 2).set_title('Rectified EMG')
        plt.plot(time, emg_rectified)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        # plt.ylim(-1.5, 1.5)
        plt.xlabel('Time (sec)')
        plt.ylabel('EMG (a.u.)')

        fig.tight_layout()
        fig_name = 'fig4.png'
        fig.set_size_inches(w=11, h=7)
        fig.savefig(fig_name)

    def alltogether(self, time, emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
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
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.subplot(1, 3, 1).set_title('Unfiltered,' + '\n' + 'unrectified EMG')
        plt.plot(time, emg)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        #plt.ylim(-1.5, 1.5)
        plt.xlabel('Time (sec)')
        plt.ylabel('EMG (a.u.)')

        plt.subplot(1, 3, 2)
        plt.subplot(1, 3, 2).set_title(
            'Filtered,' + '\n' + 'rectified EMG: ' + str(int(high_band * sfreq)) + '-' + str(int(low_band * sfreq)) + 'Hz')
        plt.plot(time, emg_rectified)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        #plt.ylim(-1.5, 1.5)
        #plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
        plt.xlabel('Time (sec)')

        plt.subplot(1, 3, 3)
        plt.subplot(1, 3, 3).set_title(
            'Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(int(low_pass * sfreq)) + ' Hz')
        plt.plot(time, emg_envelope)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        #plt.ylim(-1.5, 1.5)
        #plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
        plt.xlabel('Time (sec)')

        plt.subplot(1, 4, 4)
        plt.subplot(1, 4, 4).set_title('Focussed region')
        plt.plot(time[int(0.9 * 1000):int(1.0 * 1000)], emg_envelope[int(0.9 * 1000):int(1.0 * 1000)])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.xlim(0.9, 1.0)
        plt.ylim(-1.5, 1.5)
        plt.xlabel('Time (sec)')

        fig_name = 'fig_' + str(int(low_pass * sfreq)) + '.png'
        fig.set_size_inches(w=11, h=7)
        fig.savefig(fig_name)
        return emg_envelope, emg_filtered, emg_rectified

    def integratedEMG(self, array):
        n = array.size
        iemg = 0
        for i in range(0, n):
            iemg += abs(array[i])
        return iemg

    def rmsValue(self, array):
        n = array.size
        squre = 0.0

        # calculating Square
        for i in range(0, n):
            squre += (array[i] ** 2)
        # Calculating Mean
        mean = (squre / (float)(n))
        # Calculating Root
        root = math.sqrt(mean)
        return root

    def wavLen(self, array):  # Waveform length
        n = array.size
        wl = 0
        for i in range(0, n - 1):
            wl += abs(array[i + 1] - array[i])
        return wl

    def aregression(self, array):
        # train autoregression
        model = AutoReg(array, lags=3)
        model_fit = model.fit()
        # print('Coefficients: %s' % model_fit.params)
        return model_fit.params

    def mavValue2(self, array):
        n = array.size
        mmav2 = 0.0
        for i in range(0, n):
            if (0.25 * n > i):
                w = 4 * i / n
            elif (0.75 * n < i):
                w = 4 * (i - n) / n
            else:
                w = 1
            mmav2 += w * abs(array[i])
        mmav2 = mmav2 / n
        return mmav2

    def ssIntegral(self, array):  # Simple Square Integral
        n = array.size
        ssi = 0.0
        for i in range(0, n):
            ssi += abs(array[i]) ** 2
        return ssi

    def variance(self, array):
        n = array.size
        var = 0.0
        for i in range(0, n):
            var += array[i] ** 2
        var = var / (n - 1)
        return var

    def standardDev(self, array):
        std = np.std(array)
        return std

    def numofPeaks(self, array):
        n = array.size
        nop = 0
        mean_list = []
        rms = self.rmsValue(array)
        for i in range(0, n):
            if array[i] > rms:
                mean_list = np.append(mean_list, array[i])
                nop += 1
        mean = np.mean(mean_list)
        return nop, mean

    def dumvValue(self, array):  # Difference absolute mean value
        n = array.size
        dumv = 0.0
        for i in range(0, n - 1):
            dumv += abs(array[i + 1] - array[i])
        dumv = dumv / n
        return dumv

    def mflValue(self, array):  # Maximum fractal length
        n = array.size
        mfl = 0.0
        for i in range(0, n - 1):
            mfl += (array[i + 1] - array[i]) ** 2
        mfl = math.log10(math.sqrt(mfl))
        return mfl

    def percent(self, array):  # Percentile (Perc)
        perc = np.percentile(array, 75)
        return perc

    def mValue(self, array, k):  # Calculates M value for skew and kurt
        n = array.size
        m = 0.0
        mean = np.mean(array)
        for i in range(0, n):
            m += (array[i] - mean) ** k
        m = m / n
        return m

    def skewness(self, array):  # Skewness (Skew)
        m2 = self.mValue(array, 2)
        m3 = self.mValue(array, 3)
        skew = m3 / (m2 * math.sqrt(m2))
        return skew

    def kurtosis(self, array):  # Kurtosis (Kurt)
        m2 = self.mValue(array, 2)
        m4 = self.mValue(array, 4)
        kurt = m4 / (m2 * m2)
        return kurt

    def zeroCrossing(self, array):  # Zero crossing (ZC)
        n = array.size
        zc = 0
        for i in range(1, n):
            if -(array[i]) * array[i - 1] > 0:
                zc += 1
        return zc

    def sscValue(self, array):  # Slope sign changes (SSC)
        n = array.size
        ssc = 0
        tr = 100  # threshold
        for i in range(1, n - 1):
            f = (array[i] - array[i - 1]) * (array[i] - array[i + 1])
            if f > tr:
                ssc += 1
        return ssc

    def wilsonAmplitude(self, array):
        n = array.size
        wa = 0
        tr = 100
        for i in range(0, n - 1):
            f = abs(array[i] - array[i + 1])
            if f > tr:
                wa += 1
        return wa

    def window(self, emg):
        rmsValueAll, wlValueAll, aregValueAll1, aregValueAll2, \
        aregValueAll3, aregValueAll4, iemgValueAll, mav2ValueAll, \
        ssiValueAll, varValueAll, stdevValueAll, nopValueAll, mnopValueAll, \
        dumvValueAll, mflValueAll, percentValueAll, skewValueAll, \
        kurtValueAll, zcValueAll, sscValueAll, waValueAll = (np.empty([0]),) * 21
        win = 100  # the size of a single window
        shift = 50  # the step of shift
        for i in range(0, emg.size // shift - 1):
            new_emg = emg[i * shift:i * shift + win]
            # new_emg_filt = emg_filtred[i*shift:i*shift+win]
            rmsValueAll = np.append(rmsValueAll, self.rmsValue(new_emg))
            wlValueAll = np.append(wlValueAll, self.wavLen(new_emg))
            coefArr = self.aregression(new_emg)
            aregValueAll1 = np.append(aregValueAll1, coefArr[0])
            aregValueAll2 = np.append(aregValueAll2, coefArr[1])
            aregValueAll3 = np.append(aregValueAll3, coefArr[2])
            aregValueAll4 = np.append(aregValueAll4, coefArr[3])
            iemgValueAll = np.append(iemgValueAll, self.integratedEMG(new_emg))
            mav2ValueAll = np.append(mav2ValueAll, self.mavValue2(new_emg))
            ssiValueAll = np.append(ssiValueAll, self.ssIntegral(new_emg))
            varValueAll = np.append(varValueAll, self.variance(new_emg))
            stdevValueAll = np.append(stdevValueAll, self.standardDev(new_emg))
            nopValueAll1, mnopValueAll1 = self.numofPeaks(new_emg)
            nopValueAll = np.append(nopValueAll, nopValueAll1)
            mnopValueAll = np.append(mnopValueAll, mnopValueAll1)
            dumvValueAll = np.append(dumvValueAll, self.dumvValue(new_emg))
            mflValueAll = np.append(mflValueAll, self.mflValue(new_emg))
            percentValueAll = np.append(percentValueAll, self.percent(new_emg))
            skewValueAll = np.append(skewValueAll, self.skewness(new_emg))
            kurtValueAll = np.append(kurtValueAll, self.kurtosis(new_emg))
            zcValueAll = np.append(zcValueAll, self.zeroCrossing(new_emg))
            sscValueAll = np.append(sscValueAll, self.sscValue(new_emg))
            waValueAll = np.append(waValueAll, self.wilsonAmplitude(new_emg))

        emg_features = (np.vstack((rmsValueAll, wlValueAll, aregValueAll1, aregValueAll2,
                                   aregValueAll3, aregValueAll4, iemgValueAll, mav2ValueAll,
                                   ssiValueAll, varValueAll, stdevValueAll, nopValueAll, mnopValueAll,
                                   dumvValueAll, mflValueAll, percentValueAll, skewValueAll,
                                   kurtValueAll, zcValueAll, sscValueAll, waValueAll))).T
        return emg_features

    def dif_slopes(self, emg):
        zcValueAll, sscValueAll, waValueAll = (np.empty([0]),) * 3
        win = 100  # the size of a single window
        shift = 50  # the step of shift
        for i in range(0, emg.size // shift - 1):
            new_emg = emg[i * shift:i * shift + win]
            zcValueAll = np.append(zcValueAll, self.zeroCrossing(new_emg))
            sscValueAll = np.append(sscValueAll, self.sscValue(new_emg))
            waValueAll = np.append(waValueAll, self.wilsonAmplitude(new_emg))

        return zcValueAll, sscValueAll, waValueAll