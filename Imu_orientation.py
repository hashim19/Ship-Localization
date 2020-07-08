import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

import localization_utils as lu
from Orient_KF import KalmanFilter
import config as cfg


def accel_mag_orient(accel, mag=None):
    """
    Compute orientation i.e. roll, pitch and yaw from accelerometer and magnetometer.

    accel: A numpy array of accel data. Arranged as 3xR, where R is the number of data points and 3 is for x,y and z axis.

    mag: A numpy array of accel data. Arranged as 2xR, where R is the number of data points and 3 is for x,y axis. Default value is None.
    """

    # accel = accel * (156.9/(2**15))

    accel = accel / np.linalg.norm(accel)
    
    accel_x = accel[0]
    accel_y = accel[1]
    accel_z = accel[2]

    # compute roll and pitch
    if accel_z == 0.0:
        roll = 0.0
    else:
    # roll = np.rad2deg(math.atan2(accel_y, accel_z))
        roll = np.rad2deg(math.atan2(accel_y, np.sqrt(accel_x**2 + accel_z**2) ) )

    if accel_y == 0 and accel_z == 0:
        pitch = 0.0
    else:
        pitch = np.rad2deg(math.atan2(accel_x, np.sqrt(accel_y**2 + accel_z**2) ) )

    if mag is not None:
        # mag = mag / np.linalg.norm(mag)
        # print(mag)
        p = np.deg2rad(pitch)
        r = np.deg2rad(roll)

        mag_x = mag[0]*np.cos(p) + mag[1]*np.sin(r)*np.sin(p) + mag[2]*np.cos(r)*np.sin(p)
        mag_y = mag[1] * np.cos(r) - mag[2] * np.sin(r)

        # mag_x = mag[0]*(1-accel_x**2) - mag[1]*(accel_x*accel_y) - mag[2]*accel_x*(np.sqrt(1 - accel_x**2 - accel_y**2))
        # mag_y = mag[1]*(np.sqrt(1 - accel_x**2 - accel_y**2)) - mag[2]*accel_y

        yaw = np.rad2deg(math.atan2(mag_x, mag_y))

        # mag_x = mag[0]
        # mag_y = mag[1]

        # yaw = np.rad2deg(math.atan2(-mag_y, mag_x))

        # print(mag_x)
        # print(mag_y)

        # mag_x = 22.720337
        # mag_y = -22.923279

        # if mag_x < 0:
        #     yaw = 180 - (math.atan2(mag_y,mag_x))*(180/np.pi)
        # if mag_x > 0 and mag_y < 0:
        #     yaw = - (math.atan2(mag_y,mag_x))*(180/np.pi)
        # if mag_x > 0 and mag_y > 0:
        #     yaw = 360 - (math.atan2(mag_y,mag_x))*(180/np.pi)
        # if mag_x == 0 and mag_y < 0:
        #     yaw = 90
        # if mag_x == 0 and mag_y > 0:
        #     yaw = 270    

        # if mag_y > 0:
        #     yaw = (math.atan2(mag_x,mag_y))*(180/np.pi)
        # if mag_y < 0:
        #     yaw = 270 - (math.atan2(mag_x,mag_y))*(180/np.pi)
        # if mag_y == 0 and mag_x < 0:
        #     yaw = 180
        # if mag_y == 0 and mag_x > 0:
        #     yaw = 0
    else:
        yaw = 0

    return np.array([roll, pitch, yaw])


def init_orient(accel, N=10, mag=None):
    """
    Compute initial orientation by averaging first n data points of accelerometer and magnetometer.

    accel: A numpy array of accel data. Arranged as 3xR, where R is the number of data points and 3 is for x,y and z axis.

    N: number of points to average in the data. Starts from 0 and take average of first N data points along each dimension. Default to 10

    mag: A numpy array of accel data. Arranged as 2xR, where R is the number of data points and 3 is for x,y axis. Default value is None.
    """

    # print(accel[0,:10])
    # print(accel[1,:10])

    # averaging accel
    accel_avg_x = np.average(accel[0,:N])
    accel_avg_y = np.average(accel[1,:N])
    accel_avg_z = np.average(accel[2,:N])

    accel_avg = np.array([accel_avg_x, accel_avg_y, accel_avg_z])
    # print(accel_avg)

    # averaging mag
    mag_avg = None
    if mag is not None:
        mag_avg_x = np.average(mag[0,:N])
        mag_avg_y = np.average(mag[1,:N])
        mag_avg_z = np.average(mag[2,:N])

        mag_avg = np.array([mag_avg_x, mag_avg_y, mag_avg_z])
        # mag_avg = mag[:, 0]

    init_angle = accel_mag_orient(accel_avg, mag=mag_avg)

    return init_angle


def gyro_orient(R_in, gyro, dt=0.02):

    # integrate gyro
    w = gyro*(dt)
    w_norm = np.linalg.norm(w)

    if w_norm > 0:
        sk_w = lu.SkewMat((w/w_norm).reshape(3,1))

        R_exp = np.identity(3) + (np.sin(w_norm))*(sk_w) + (1-np.cos(w_norm))*(np.matmul(sk_w,sk_w))

    else:
        R_exp = np.identity(3)

    # print(R_exp)
    R_out = np.matmul(R_in, R_exp)

    return R_out

def create_kf_orient(init_x, kf_cfg, dt=0.02):

    # initialize kalman filter
    if kf_cfg.orient_settings['mag']:
        kf = KalmanFilter(dim_x=6, dim_z=3)
    else:
        kf = KalmanFilter(dim_x=6, dim_z=2)
    
    # initial nav state
    kf.x = init_x

    # state transition matrix
    kf.A = np.array([[1.0, 0, 0, -dt, 0, 0],
                    [0.0, 1.0, 0, 0, -dt, 0],
                    [0.0, 0.0, 1.0, 0, 0, -dt,],
                    [0.0, 0.0, 0.0, 1.0, 0, 0],
                    [0.0, 0.0, 0.0, 0, 1.0, 0],
                    [0.0, 0.0, 0.0, 0, 0, 1.0]])

    # measurement model
    kf.H = np.array([[1, 0.0, 0.0, 0.0, 0, 0],
                    [0.0, 1, 0.0, 0.0, 0, 0],
                    [0.0, 0, 1.0, 0.0, 0, 0]])

    # measurement covariance
    kf.R = np.diag(kf_cfg.orient_measurement['sigma_accel_mag'])**2 

    if kf_cfg.orient_settings['mag']:
        kf.H = kf.H
        kf.R = kf.R
    else:
        kf.H = kf.H[:2, :]
        kf.R = kf.R[:2, :2]

    # state covariance
    kf.P = np.diag(kf_cfg.orient_factp)**2

    # process covariance
    kf.Q = np.zeros((6,6))
    kf.Q[0:3,0:3] = np.diag(kf_cfg.orient_process['sigma_gyro'])**2
    kf.Q[3:6,3:6] = np.diag(kf_cfg.orient_process['sigma_gyro_bias'])**2 

    # control matrix
    kf.B = np.array([
        [dt, 0., 0],
        [0, dt, 0],
        [0, 0, dt],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    return kf



def fuse_gyro_accel_mag(gyro_orient, accel_mag_orient, kf):

    # prediction step
    kf.predict(u=gyro_orient)

    # update step
    kf.update(accel_mag_orient)

    return kf


if __name__ == '__main__':

    # filename = 'Lums_Cricket_Ground_Square.csv'
    filename = './Data2/nlzdr.csv'

    imu_df = pd.read_csv(filename)

    print(imu_df)

    # accel_x = imu_df['BMI120_Accelerometer.x'].to_numpy()
    # accel_y = imu_df['BMI120_Accelerometer.y'].to_numpy()
    # accel_z = imu_df['BMI120_Accelerometer.z'].to_numpy()

    accel = imu_data[["AccX", 'AccY', 'AccZ']].to_numpy().T

    # print(accel_x)

    # accel = np.block([
    #     [accel_x],
    #     [accel_y],
    #     [accel_z]
    # ])

    print(accel)
    print(np.shape(accel))

    ####### initial orientation ########

    # accel_init = np.block([
    #     [accel_x],
    #     [accel_y],
    #     [accel_z]
    # ])

    init_rpy = init_orient(accel)

    print(init_rpy)
    init_state = np.hstack((init_rpy, np.zeros(3)))
    print(init_state)
    Kalman_Kf = create_kf_orient(init_state, cfg) 

    print(Kalman_Kf.A)
    print(Kalman_Kf.H)
    print(Kalman_Kf.R)
    print(Kalman_Kf.P)
    print(Kalman_Kf.Q)
    print(Kalman_Kf.B)

    # ########### angles from accel and mag #########
    rpy_ls = []
    rpy_gyro_ls = []
    R_nb = R.from_euler('xyz', init_rpy, degrees=True).as_matrix()
    # print(R_nb)

    for idx, row in imu_df.iterrows():
    # # for acc in accel.T:
    #     # print(acc)

        ind = np.zeros(3)
        ind[0:2] = 1
        
        if cfg.orient_settings['mag']:
            ind[2] = 1
        
        acc = row[1:4].to_numpy()
        gyro = np.rad2deg(row[4:7].to_numpy())

    #     # print(acc)

        R_nb = gyro_orient(R_nb, np.deg2rad(gyro))

        rpy_gyro = R.from_matrix(R_nb).as_euler('xyz', degrees=True)
    #     # rpy = accel_mag_orient(filtered_accel)
        rpy_accel = accel_mag_orient(acc)

        ind = ind == 1

        rpy_accel = rpy_accel[ind]
        # Kalman_Kf.H = Kalman_Kf.H[ind, :]
        # Kalman_Kf.R = Kalman_Kf.R[ind,:]
        # Kalman_Kf.R = Kalman_Kf.R[:, ind]

        Kalman_Kf = fuse_gyro_accel_mag(gyro, rpy_accel, Kalman_Kf)

        # print(Kalman_Kf.x)

    #     # print(rpy)
        rpy = Kalman_Kf.x[:3]
        rpy_ls.append(rpy)
        rpy_gyro_ls.append(rpy_gyro)

    rpy = np.array(rpy_ls).T
    rpy_gyro = np.array(rpy_gyro_ls).T

    plt.figure(figsize=(12,9))
    plt.plot(rpy[0, :], label='Roll kf')
    plt.plot(rpy[1, :], label='Pitch kf')
    plt.plot(rpy[2, :], label='Yaw kf')

    plt.plot(rpy_gyro[0, :], label='Roll gyro')
    plt.plot(rpy_gyro[1, :], label='Pitch gyro')
    plt.plot(rpy_gyro[2, :], label='Yaw gyro')
    
    plt.grid()
    plt.legend()
    plt.show()


