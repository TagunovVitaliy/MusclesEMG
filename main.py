import serial
import re
import pyqtgraph as pg
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket
import filter as ft

# Max amount of points displaying on the chart
vals_limit = 100

# Create figure for plotting
# xs = np.arange(vals_limit)
# ys = np.zeros(vals_limit)
# ds = np.empty([0, 2])
# pw = pg.plot()
# curve = pw.plot(xs, ys, title='EMG over time')
# pw.setLabel('left', 'Power')

# Specify the port name
# You should check this and change (if need) each time you reconnect the Arduino part
port1 ='/dev/tty.HC-06-DevB'
port2 ='/dev/tty.HC-06-DevB-1'
# HC06 defaults to this value
# You should not change this constant
baudrate = 9600

# Connect to EMG serial port
ser1 = serial.Serial(port1, baudrate)
# Connect to RoboArm serial port
ser2 = serial.Serial(port2, baudrate)

# Determ actual time
start_time = time.monotonic()

def read_emg_once():
    # Read one line
    line = ser1.readline()
    # Parse the line to extract the signal value
    signal_search = re.search('([0-9]+)', line.decode('utf-8'))
    if signal_search:
        signal = int(signal_search.group(1)) #for realtime signal = int(signal_search.group(1))
        # print(signal)
        # f = open('ecr_r5.txt', 'a')
        t = round(time.monotonic() - start_time, 3)
        # f.write(signal + "," + str(t) + "\n")
        # f.close()

        signal_time = np.array([signal, t])
        # num_rows = np.shape(signal_time)
        # print(num_rows, '\n')
        # print(signal_time, '\n')
        return signal_time
    return None

def write_to_robot(move):
    ser2.write(bytes(str(move), 'utf-8'))

# This function is called periodically from FuncAnimation
def update():

    global ys, xs, ds
    # Read EMG in each animate() call
    emg_signal = read_emg_once()
   # print(emg_signal, '\n')
    if not emg_signal[0]:
        return

    # Add x and y to lists
    ys = np.append(ys, emg_signal[0])
    ds = np.vstack([ds, emg_signal])

    # Limit x and y lists to 20 items
    # xs = xs[-vals_limit:]
    # ys = ys[-vals_limit:]

    # Draw x and y lists
    #curve.setData(xs, ys)

    # Sending new dataset to processing
    num_rows = np.shape(ds)[0]
    if num_rows >= 250:
        move = ft.data_processing(ds)
        write_to_robot(move)
        ds = np.empty([0, 2])

# Set up calling update() function periodically
# Reading emg is performed inside update()
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.setInterval(1)
timer.start()
#pg.QtCore.QTimer.singleShot(10 * 1000, pg.QtCore.QCoreApplication.quit)

# Following lines are required for pyqtgraph
# (besacuse it is based on Qt and we have to start QApplication to show any visual)
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()