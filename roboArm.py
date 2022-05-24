# import serial
# import time
#
# port2 ='/dev/tty.HC-06-DevB-1'
# baudrate = 9600
#
# arduino = serial.Serial(port2, baudrate)
# time.sleep(2)
# # def write_read(x):
# #     arduino.write(bytes(x, 'utf-8'))
# #     data = arduino.readline()
# #     return data
# # while True:
# #     num = input("Enter a number: ") # Taking input from user
# #     value = write_read(num)
# #     print(value) # printing the value
# while (1):
#     dataFromUser = input()
#     if dataFromUser == '1':
#         arduino.write(bytes('1', 'utf-8'))
#     elif dataFromUser == '0':
#         arduino.write(bytes('0', 'utf-8'))
