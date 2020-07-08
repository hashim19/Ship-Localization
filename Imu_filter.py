#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math


#imu filter function
def imu_filter(x, filter='butter', filter_order=2, cut_off_freq=10, filter_type='low', sampling_freq=20):

    """
    x: input array to be filtered
    
    filter: use 'butter' for butterworth filter. Only implemented butterworth filter for now. It is possible to implement more.
    
    filter_order: order of the butterworth flter. 

    cut_off_freq: cut off frequency for filtering. If filter_type is low (lowpass), the filter function filters all the frequencies greater than 
                    cut_off_freq and if filter_type is hp (highpass), the function filters all frequencies below the cut_off_freq. cut_off_freq 
                    must be less than half the sampling_freq.

    sampling_freq: sampling freq of the imu.
    """

    Wn = cut_off_freq / (0.5*sampling_freq)
    if filter=='butter':
        sos = signal.butter(filter_order, Wn, filter_type, fs=sampling_freq, output='sos')
        
        y = signal.sosfiltfilt(sos, x)

        return y


def imu_LowPass(xt, yt_minus_1, alpha):

    """
    xt: input data array to be filtered

    yt_minus_1: previous filtered signal

    alpha: smoothing factor

    """
    if yt_minus_1 is not None:
        yt = alpha*xt + (1-alpha)*yt_minus_1
    else:
        return xt

    return yt


if __name__ == '__main__':

    filename = 'Lums_Cricket_Ground_Square.csv'

    imu_df = pd.read_csv(filename)

    print(imu_df)

    accel_x = imu_df['BMI120_Accelerometer.x'].to_numpy()
    accel_y = imu_df['BMI120_Accelerometer.y'].to_numpy()
    accel_z = imu_df['BMI120_Accelerometer.z'].to_numpy()

    # print(accel_x)

    ######## butterworth ########
    accel_x_filtered = imu_filter(accel_x, cut_off_freq=9, filter_type='low')
    accel_y_filtered = imu_filter(accel_y, cut_off_freq=9, filter_type='low')
    accel_z_filtered = imu_filter(accel_z, cut_off_freq=9, filter_type='low')

    accel = np.block([
        [accel_x_filtered],
        [accel_y_filtered],
        [accel_z_filtered]
    ])

    print(accel)
    print(np.shape(accel))

    # print(accel_x_filtered)

    ########## low pass ##########
    accel_ls = []
    t = 0
    filtered_accel = np.array([0,0,0])
    for acc in accel.T:

        filtered_accel = imu_LowPass(acc, filtered_accel, 0.5)

        accel_ls.append(filtered_accel)

    accel_ls = np.array(accel_ls).T
    print(accel_ls)
    plt.figure(figsize=(12,9))

    plt.plot(accel_x, label='original')
    plt.plot(accel_ls[0,:], label='filtered_lowpass')
    plt.plot(accel_x_filtered, label='filtered_butter')
    plt.grid()
    plt.legend()

    plt.title('Filtering along X-axis')

    plt.show()


