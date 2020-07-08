"""
This is the main script which you will usually use. This script runs in the following steps:

1). read the data from sensors

2). filter the data

3). compute orientation from the sensors

4). compute travelled distance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

# import localization_utils as lu
from KF import KalmanFilter
import config as cfg
from Imu_filter import imu_filter, imu_LowPass
import Imu_orientation as ImO 
from imu_position import integrate_accel
import localization_utils as lu

# read the data 

folder = './Data2/'
filename = 'Car_Long'
fullfile = folder + filename + '.csv'
# filename = './Data2/Ground_Square.csv'

if filename == 'Ground_Square':
    
    # read data
    imu_data = pd.read_csv(fullfile)

    # extract accel data
    accel = imu_data[['BMI120_Accelerometer.x', 'BMI120_Accelerometer.y', 'BMI120_Accelerometer.z']].to_numpy().T
    mag = None
    
    # time period
    # dt = np.round(imu_data['timestamp'].iloc[1] - imu_data['timestamp'].iloc[0], 2) /1000
    dt = 0.02

elif filename == 'Car_Long':
    
    # read data
    imu_data = pd.read_csv(fullfile, skiprows = [1,2])

    # extract accel and mag data
    accel = imu_data[['BMI120_Accelerometer.x', 'BMI120_Accelerometer.y', 'BMI120_Accelerometer.z']].to_numpy().T
    mag = imu_data[["AK09918_Magnetometer.x", 'AK09918_Magnetometer.y', 'AK09918_Magnetometer.z']].to_numpy().T

    # time period
    # dt = np.round(imu_data['timestamp'].iloc[1] - imu_data['timestamp'].iloc[0], 2) /1000
    dt = 0.01

elif filename == 'sensorMotSim':

    # read data
    imu_data = pd.read_csv(fullfile)
    
    # rename columns
    imu_data = imu_data.rename(columns={"Acc X": "AccX", "Acc Y": "AccY", "Acc Z": "AccZ", "Ang Vel X": "GyrX", "Ang Vel Y": "GyrY", "Ang Vel Z": "GyrZ"})
    imu_data = imu_data[['Time', "AccX", 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']]

    # extract accel and mag data
    accel = imu_data[["AccX", 'AccY', 'AccZ']].to_numpy().T
    # mag = imu_data[["Mag(1)", 'Mag(2)', 'Mag(3)']].to_numpy().T

    # time period
    # pattern = '%Y-%m-%d %H:%M:%S.%f'

    # imu_time1 = datetime.strptime(imu_data['Time'].iloc[0], pattern)
    # imu_time2 = datetime.strptime(imu_data['Time'].iloc[1], pattern)

    dt = np.round(imu_data['Time'].iloc[2] - imu_data['Time'].iloc[1], 3)
    # dt = imu_data['Time'].iloc[1]
    print(filename)

else:
    # read data
    imu_data = pd.read_csv(fullfile)

    # extract accel and mag data
    accel = imu_data[["AccX", 'AccY', 'AccZ']].to_numpy().T
    mag = imu_data[["Mag(1)", 'Mag(2)', 'Mag(3)']].to_numpy().T

    # time period
    pattern = '%Y-%m-%d %H:%M:%S.%f'

    imu_time1 = datetime.strptime(imu_data['Time'].iloc[0], pattern)
    imu_time2 = datetime.strptime(imu_data['Time'].iloc[1], pattern)

    dt = np.round(imu_time2.timestamp() - imu_time1.timestamp(), 2)


print(imu_data)

print(dt) 
fs = 1/dt
print(fs)

# apply butterworth filter
if cfg.settings['use_butterworth']:

    # change these cut-off frequencies for different results
    accel_cut_off = 8
    mag_cut_off = 5

    ############# filtering acceleromter data ###############
    accel_x = imu_data["AccX"].to_numpy().copy()
    accel_y = imu_data['AccY'].to_numpy().copy()
    accel_z = imu_data['AccZ'].to_numpy().copy()

    # change cut-off frequency accordingly. It must be less than half the sampling frequency. See docs of imu_filter
    accel_x_filtered = imu_filter(accel_x, cut_off_freq=accel_cut_off, filter_type='low', sampling_freq=fs)
    accel_y_filtered = imu_filter(accel_y, cut_off_freq=accel_cut_off, filter_type='low', sampling_freq=fs)
    accel_z_filtered = imu_filter(accel_z, cut_off_freq=accel_cut_off, filter_type='low', sampling_freq=fs)

    accel_filtered = np.block([
        [accel_x_filtered],
        [accel_y_filtered],
        [accel_z_filtered]
    ])

    # save back to imu data frame
    imu_data['AccX'] = accel_x_filtered
    imu_data['AccY'] = accel_y_filtered
    imu_data['AccZ'] = accel_z_filtered

    # plot original/filtered data
    plt.figure(figsize=(12,9))
    plt.plot(accel_x, label='original')
    # plt.plot(accel_ls[0,:], label='filtered_lowpass')
    plt.plot(accel_x_filtered, label='filtered_butter')
    plt.grid()
    plt.legend()

    plt.title('Filtering along X-axis')

    ################# filtering magnetometer data #################
    mag_x = imu_data["Mag(1)"].to_numpy().copy()
    mag_y = imu_data['Mag(2)'].to_numpy().copy()
    mag_z = imu_data['Mag(3)'].to_numpy().copy()

    # change cut-off frequency accordingly. It must be less than half the sampling frequency. See docs of imu_filter
    mag_x_filtered = imu_filter(mag_x, cut_off_freq=mag_cut_off, filter_type='low', sampling_freq=fs)
    mag_y_filtered = imu_filter(mag_y, cut_off_freq=mag_cut_off, filter_type='low', sampling_freq=fs)
    mag_z_filtered = imu_filter(mag_z, cut_off_freq=mag_cut_off, filter_type='low', sampling_freq=fs)

    mag_filtered = np.block([
        [mag_x_filtered],
        [mag_y_filtered],
        [mag_z_filtered]
    ])

    # save back to imu data frame
    imu_data['Mag(1)'] = mag_x_filtered
    imu_data['Mag(2)'] = mag_y_filtered
    imu_data['Mag(3)'] = mag_z_filtered

    # plot original/filtered data
    plt.figure(figsize=(12,9))
    plt.plot(mag_x, label='original')
    # plt.plot(accel_ls[0,:], label='filtered_lowpass')
    plt.plot(mag_x_filtered, label='filtered_butter')
    plt.grid()
    plt.legend()

    plt.title('Filtering Magnetometer along X-axis')

    plt.show()

# compute initial orientation for kalman filter
if cfg.orient_settings['mag']:
    init_orient = ImO.init_orient(accel, mag=mag)
else:
    init_orient = ImO.init_orient(accel)

init_orient = np.hstack((init_orient, np.zeros(3)))
# init_orient = np.hstack((np.zeros(3), np.zeros(3)))

print(init_orient)

R_nb = R.from_euler('xyz', init_orient[:3], degrees=True).as_matrix()
Rz = lu.Rz(np.deg2rad(-90))
R_nb = np.matmul(Rz, R_nb)
init_orient[:3] = R.from_matrix(R_nb).as_euler('xyz', degrees=True)

# create orientation kalman filter object
kf_orient = ImO.create_kf_orient(init_orient, cfg, dt=dt)
print(kf_orient.x)
print(kf_orient.z)

########### angles from accel and mag #########
rpy_ls = []
rpy_accel_ls = []
rpy_gyro_ls = []
pos_ls = []

p_xyz = np.array([0., 0, 0])
v_xyz = np.array([0., 0, 0])
# filtered_mag = init_orient[:3]
filtered_accel = np.array([0,0,0])

t_fuse = 0

accel_bias = np.array([0.239914, -0.1748889, 0])

for idx, row in imu_data.iterrows():
    
    # define indices of len 3x1 
    ind = np.zeros(3)
    ind[0:2] = 1 # ind = [1, 1, 0]
    
    if cfg.orient_settings['mag']:
        ind[2] = 1  # ind = [1, 1, 1]
    
    # get accel, gyo , mag 
    if filename == 'Ground_Square' or filename == 'Car_Long' or filename == 'sensorMotSim':

        # accel
        acc = row[1:4].to_numpy().astype(np.float) - accel_bias
        # acc[0] = 0

        # filter accel
        filtered_accel = imu_LowPass(acc, filtered_accel, 0.6)

        # gyro and mag
        if filename == 'Car_Long':
            gyro = -row[7:10].to_numpy()
            magneto = np.matmul(Rz,row[4:7].to_numpy())
            
            # filter magnetometer data
            if idx == 0:
                filtered_mag = magneto
            filtered_mag = imu_LowPass(magneto, filtered_mag, 0.6)

        else:
            # magnetometer not in Ground_Square and sensorMotSim
            gyro = row[4:7].to_numpy().astype(np.float)
            magneto = None
    
    else:
        
        acc = row[1:4].to_numpy().astype(np.float) * 9.81

        # filter accel
        filtered_accel = imu_LowPass(acc, filtered_accel, 0.6)

        # get gyro and mag
        gyro = row[4:7].to_numpy().astype(np.float)
        magneto = row[7:10].to_numpy() * 0.01

        # filter magnetometer data
        if idx == 0:
            filtered_mag = magneto
        filtered_mag = imu_LowPass(magneto, filtered_mag, 0.6)

    # acc = row[1:4].to_numpy().astype(np.float)
    # filtered_accel = imu_LowPass(acc, filtered_accel, 0.6)
    # print(acc)
    # gyro = np.rad2deg(row[4:7].to_numpy().astype(np.float))
    # gyro = row[4:7].to_numpy().astype(np.float)
    # gyro = np.rad2deg(row[4:7].to_numpy())

    # car long 
    # gyro = -row[7:10].to_numpy()
    # print(gyro)
    # magneto = row[7:10].to_numpy() * 0.01

    if idx != 0:

        if filename == 'Ground_Square' or filename == 'Car_Long':

            dt = np.round(imu_data['timestamp'].iloc[idx] - imu_data['timestamp'].iloc[idx-1], 2)/1000
        else:
            
            if filename == 'sensorMotSim':
                dt = np.round(imu_data['Time'].iloc[idx] - imu_data['Time'].iloc[idx-1], 3)
                # print(dt)
            else:
                imu_time1 = datetime.strptime(imu_data['Time'].iloc[idx-1], pattern)
                imu_time2 = datetime.strptime(imu_data['Time'].iloc[idx], pattern)

                dt = np.round(imu_time2.timestamp() - imu_time1.timestamp(), 2)

    # R_nb = ImO.gyro_orient(R_nb, np.deg2rad(gyro))
    
    # dt = 0.01
    
    R_nb = ImO.gyro_orient(R_nb, gyro, dt=dt)

    rpy_gyro = R.from_matrix(R_nb).as_euler('xyz', degrees=True)

    # filtered version
    if cfg.orient_settings['mag']:
        rpy_accel = ImO.accel_mag_orient(filtered_accel, mag=filtered_mag)
    else:
        rpy_accel = ImO.accel_mag_orient(acc)
    # print(rpy_accel)
    # print(rpy_gyro)

    ind = ind == 1
    # print(ind)
    # rpy_accel = rpy_accel[ind]
    # print(rpy_accel)
    # Kalman_Kf.H = Kalman_Kf.H[ind, :]
    # Kalman_Kf.R = Kalman_Kf.R[ind,:]
    # Kalman_Kf.R = Kalman_Kf.R[:, ind]

    # if t_fuse >= 0.5:

    kf_orient = ImO.fuse_gyro_accel_mag(np.rad2deg(gyro), rpy_accel, kf_orient)
        # t_fuse = 0
    # else:
        # kf_orient.x[:3] = rpy_gyro
        # rpy_accel_avg = ImO.accel_mag_orient(acc)

    # kf_orient = ImO.fuse_gyro_accel_mag(gyro, rpy_accel, kf_orient)
    # print(kf_orient.x[:3])

    # compute travelled distance
    # print(acc)
    # p_xyz, v_xyz = integrate_accel(acc, kf_orient.x[:3], v_xyz, p_xyz)

    # filtered version
    # print(p_xyz)
    p_xyz, v_xyz = integrate_accel(filtered_accel, kf_orient.x[:3], v_xyz, p_xyz, dt = dt)

    # print(p_xyz)

    # print(Kalman_Kf.x)

    # print(rpy_gyro)
    rpy = kf_orient.x[:3]
    rpy_ls.append(rpy)
    rpy_gyro_ls.append(rpy_gyro)
    pos_ls.append(p_xyz)
    rpy_accel_ls.append(rpy_accel)

    t_fuse+=dt


# print(imu_data)

rpy = np.array(rpy_ls).T
rpy_gyro = np.array(rpy_gyro_ls).T
rpy_accel = np.array(rpy_accel_ls).T

plt.figure(figsize=(12,9))
plt.plot(rpy[0, :], label='Roll kf')
plt.plot(rpy[1, :], label='Pitch kf')
plt.plot(rpy[2, :], label='Yaw kf')

plt.plot(rpy_gyro[0, :], label='Roll gyro')
plt.plot(rpy_gyro[1, :], label='Pitch gyro')
plt.plot(rpy_gyro[2, :], label='Yaw gyro')

# uncomment this line to view roll pitch yaw from magnetometer as well
plt.plot(rpy_accel[0, :], label='Roll accel')
plt.plot(rpy_accel[1, :], label='Pitch accel')
plt.plot(rpy_accel[2, :], label='Yaw mag')

plt.grid()
plt.legend()
plt.savefig(filename + '_orientation.png', bbox_inches='tight', dpi=500)

# plot position
pos = np.array(pos_ls).T
print(pos)
plt.figure(figsize=(12,9))
plt.plot(pos[0,0], pos[1,0], 'gp', markersize = 8, label = 'Start imu')
plt.plot(pos[0,len(pos[0,:])-1], pos[1,len(pos[1,:])-1], 'rp', markersize = 8, label = 'Stop imu')

# plt.plot(pos[1,0], pos[0,0], 'gp', markersize = 8, label = 'Start imu')
# plt.plot(pos[1,len(pos[1,:])-1], pos[0,len(pos[0,:])-1], 'rp', markersize = 8, label = 'Stop imu')


plt.plot(pos[0,:], pos[1,:], 'g', linewidth=3, label = 'Imu pos')

plt.title('Position in ENU (Local Frame)', fontsize = 18, fontweight = 'bold' )
plt.xlabel('East (m)', fontsize = 16)
plt.ylabel('North (m)', fontsize = 16)
plt.grid()
plt.legend()
plt.savefig(filename + '_pos.png', bbox_inches='tight', dpi=500)

plt.show()