import serial
import re
import os
import pyqtgraph as pg
import time
import numpy as np
from Processing import Processing
from FetExtraction import Features

vals_limit = 100

# Create list for EMG data
ds = np.empty([0, 2])

# Specify the port name
# You should check this and change (if need) each time you reconnect the Arduino part
port1 ='/dev/tty.HC-06-DevB'
port2 ='/dev/tty.HC-06-DevB-1'
# HC06 defaults to this value
baudrate = 9600
num_file = 1

# Connect to EMG serial port
ser1 = serial.Serial(port1, baudrate)
# Connect to RoboArm serial port
ser2 = serial.Serial(port2, baudrate)

# Determ actual time
start_time = time.monotonic()

column_names = ['emg', 't']

class Collect:

    def update(self):
        global ds, num_file
        # Read EMG in each animate() call
        emg_signal = self.read_emg_once(num_file)

        if not emg_signal[0]:
            return

        # Add emg to list
        ds = np.vstack([ds, emg_signal])

        # Sending new dataset to processing
        num_rows = np.shape(ds)[0]
        if num_rows >= 1000:
            ds = np.empty([0, 2])
            num_file += 1
            return

    def read_emg_once(self, num_file):
        # Read one line
        line = ser1.readline()
        # Parse the line to extract the signal value
        signal_search = re.search('([0-9]+)', line.decode('utf-8'))
        if signal_search:
            signal = signal_search.group(1)
            f = open('raw_data/emg_data{}.txt'.format(num_file), 'a')
            t = round(time.monotonic() - start_time, 3)
            f.write(signal + "," + str(t) + "\n")
            f.close()

            signal_time = np.array([signal, t])
            # num_rows = np.shape(signal_time)
            return signal_time
        return None

class Robot():
    def update(self):
        global ds
        # Read EMG in each animate() call
        emg_signal = self.read_emg_once()

        if not emg_signal[0]:
            return

        # Sending new dataset to processing
        num_rows = np.shape(ds)[0]
        if num_rows >= 250:
            move = Processing().process_data(ds)
            # Writing to Robot:
            ser2.write(bytes(str(move), 'utf-8'))
            ds = np.empty([0, 2])

    def read_emg_once(self):
        # Read one line
        line = ser1.readline()
        # Parse the line to extract the signal value
        signal_search = re.search('([0-9]+)', line.decode('utf-8'))
        if signal_search:
            signal = int(signal_search.group(1))
            t = round(time.monotonic() - start_time, 3)
            signal_time = np.array([signal, t])
            return signal_time
        return None

enter = input("Please, type in 'Collect' if you want to collect data via EMG or \ntype in 'Robot' if you want to execute app in real-time mode\n")
file_names = []
# Creating necessary folders:
if not os.path.exists("processed"):
    os.makedirs("processed")
if not os.path.exists("fs_data"):
    os.makedirs("fs_data")

if enter.lower() == 'collect':
    print("Get ready to record 4 gestures")
    time.sleep(2)
    while num_file <= 4:
        print("please, tighten new muscle until you see the 'STOP' on screen")
        time.sleep(3)
        print("Start recording gesture {}".format(num_file))
        file_names.append('raw_data/emg_data{}.txt'.format(num_file))
        Collect().update()
        print('STOP')
        time.sleep(3)
    print("Recording is over, please see the results of processing:")
    opt_file_names = Processing().collect_data(file_names)
    input('Type any key to continue')
    print("See the results of the SVM model classification based on the Feature extraction:")
    time.sleep(2)
    Features().feature_sel()
elif enter.lower() == 'robot':
    print("Get ready to control the Robot")
    time.sleep(2)
    print("Lets start")
    while True:
        Robot().update()
else:
    raise Exception('Wrong command, try again')