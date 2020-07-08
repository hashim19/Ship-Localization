#!/usr/bin/env python3

"""

A configuration file which stores general configurations and filter settings for kalmanfilter

"""

import numpy as np

########################### Orientation Kalman Filter Settings #######################

orient_process = {
    'sigma_gyro' : np.deg2rad(np.array([0.0245, 0.0354, 0.0988])), # roll, pitch, yaw
    'sigma_gyro_bias' : np.deg2rad(np.array([0.003, 0.003, 0.003])) # roll bias, pitch bias, yaw bias 
}

orient_measurement = {
    'sigma_accel_mag' : np.deg2rad(np.array([0.602, 0.766, 0.2])) # roll, pitch, yaw
}

orient_factp = [np.deg2rad(1), np.deg2rad(1), np.deg2rad(1), 0.005, 0.005, 0.005]

orient_settings = {
    'mag' : True,
    'Fusion': True
}


########################### Posiition Kalman Filter Settings #######################

process = {
    'process_var' : 0.0001
}

measurement = {
    # 'accel_var' : np.array([2, 2]),
    'accel_var' : np.array([0.1, 0.1]),
    # 'pos_var' : np.array([0.05, 0.05]),
    'pos_var' : np.array([1, 1]),

    'vel_var' : np.array([10, 10])
}


########################### General Settings ###################################

settings = {
    'rfid_outage': False,
    'outage_start': 80,
    'outage_stop': 85,
    'Fusion': True,
    'use_butterworth': False,
    'butterworth_cut_off': 5,
    'sampling_freq': 20,
    'gyro_bias': np.array([0.0021, -0.0075, -0.0067]),
    'accel_bias': np.array([-0.0962, 0.0126, 0.0540]),
    'gravity': np.array([0, 0, 9.81])
}


