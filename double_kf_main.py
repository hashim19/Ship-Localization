import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import pymap3d

from KF import KalmanFilter
from Imu_filter import imu_filter
from Imu_filter import imu_LowPass
import Imu_orientation as ImO 
# import MatrixUtils as mu
from plot_func import plot_pos
# from plot_func import plot_animation
import config as cfg

filename = 'TEMP23-06-2020_20-00-18'
file_path = './Sample_Data/' + filename + '.csv'
# gps_filename = '../Data Collected/Car_Long_GNSS_ENU.csv'

# data_df = pd.read_csv(file_path, skiprows = [1,2,3])
data_df = pd.read_csv(file_path)
data_df = data_df.dropna()
# gps_data = pd.read_csv(gps_filename)

# print(gps_data)
# gps_data = gps_data.rename(columns={"t": 'time_gps', "pos_enu": "enu_x", "Unnamed: 2": "enu_y", "Unnamed: 3": "enu_z"})

print(data_df)
# print(gps_data)

# extract accel and mag data
accel = data_df[[' AccX(m/s^2)', ' AccY(m/s^2)', ' AccZ(m/s^2)']].to_numpy().T
mag = data_df[[" MagX(uT)", ' MagY(ut)', ' MagZ(uT)']].to_numpy().T

# initial gps
init_gps = data_df[[' Latitude', ' Longitude']].iloc[0].to_numpy().astype(np.float)
init_gps = np.hstack((init_gps, 168.024536))
# print(init_gps)

# time period
# pattern = '%Y-%m-%d %H:%M:%S.%f'

imu_time1 = data_df['Time'].iloc[0]
imu_time2 = data_df['Time'].iloc[1]

dt = np.round(imu_time2 - imu_time1, 2)

print(dt)

# dt = 0.01

rpy_ls = []
rpy_ls_mag = []

# initial orientation
init_orient = ImO.init_orient(accel, mag=mag, N=10)
init_orient = np.hstack((init_orient, np.zeros(3)))
print(init_orient)

R_nb = R.from_euler('xyz', init_orient[:3], degrees=True).as_matrix()

# create orientation kalman filter object
kf_orient = ImO.create_kf_orient(init_orient, cfg, dt=dt)
print(kf_orient.x)
print(kf_orient.z)

########### angles from accel and mag #########
rpy_ls = []
rpy_accel_ls = []
rpy_gyro_ls = []
pos_ls = []
kf_pos = []
gps_pos = []

vel_ls = []
accel_ls = []

p_xyz = np.array([0., 0, 0])
v_xyz = np.array([0., 0, 0])
# filtered_mag = init_orient[:3]
filtered_accel = np.array([0,0,0])

############ Position Kalman Filter ############### 

# KF Initialization
kf = KalmanFilter(dim_x=6, dim_z=4)
# dt = 0.005
kf.x = np.array([0, 0, 0, 0, 0, 0]) # location and velocity
# kf.A = np.array([[1.0, 0.0, dt, 0.0, (dt**2)/2, 0.0],
#                 [0.0, 1.0, 0.0, dt, 0.0, (dt**2)/2],
#                 [0.0, 0.0, 1.0, 0.0, dt, 0],
#                 [0.0, 0.0, 0.0, 1.0, 0, dt],
#                 [0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0, 1]])  # state transition matrix

kf.A = np.array([[1.0, 0.0, dt, 0.0, 0, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0],
                [0.0, 0.0, 1.0, 0.0, dt, 0],
                [0.0, 0.0, 0.0, 1.0, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])  # state transition matrix

kf.H = np.array([[1, 0.0, 0.0, 0.0, 0, 0],
                [0.0, 1, 0.0, 0.0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])    # Measurement function

kf.R = np.diag(np.hstack((cfg.measurement['pos_var'], cfg.measurement['accel_var'])))  
                    # measurement uncertainty

kf.P[:] = np.diag([5, 5, 10, 10, 5, 5])               # [:] makes deep copy

# sigma = process_var
kf.Q[:] = np.array([[0, 0.0, 0.0, 0.0, 0, 0],
                [0.0, 0, 0.0, 0.0, 0, 0],
                [0.0, 0, 0.0, 0.0, 0, 0],
                [0.0, 0, 0.0, 0.0, 0, 0],
                [0.0, 0.0, 0, 0.0, cfg.process['process_var'], 0],
                [0.0, 0.0, 0.0, 0, 0, cfg.process['process_var']]])


# time
t_from_start = 0.
t_ls = [t_from_start]
imu_time_ls = []
ctr_rfid_data = 0

# imu start time
start_imu = data_df['Time'].iloc[0]
# start_gps = gps_data['time_gps'].iloc[0]
# print(start_gps)

for idx, row in data_df.iterrows():

    accel = row[1:4].to_numpy().astype(np.float) - cfg.settings['accel_bias']
    # accel[0] = 0.

    # gyro = -row[7:10].to_numpy()
    gyro = row[4:7].to_numpy() - cfg.settings['gyro_bias']

    # mag = np.matmul(Rz,row[4:7].to_numpy())
    mag = row[26:].to_numpy()

    if idx == 0:
        
        filtered_mag = mag
        filtered_accel = accel
        dt = 0.01

        imu_time = 0
        gps_time = 0

        z_gps = row[22:24].to_numpy().astype(np.float)
        z_gps = np.hstack((z_gps, 168.024536))

        print(z_gps)

        z_ecef = np.array(pymap3d.geodetic2ecef( *z_gps, ell=None, deg=True))

        print(z_ecef)

        z_enu = np.array(pymap3d.ecef2enu(*z_ecef, *init_gps))[:2]

        z_enu_prev= z_enu

        # continue
    else:
        dt = np.round(data_df['Time'].iloc[idx] - data_df['Time'].iloc[idx-1], 2)
        # dt = 0.02

        imu_time = round((data_df['Time'].iloc[idx] - start_imu), 3)

        # gps_time = round((gps_data['time_gps'].iloc[ctr_rfid_data] - start_gps), 3)

    # print(dt)
    filtered_mag = imu_LowPass(mag, filtered_mag, 0.3)
    filtered_accel = imu_LowPass(accel, filtered_accel, 0.3)
    
    # orientation from gyro
    R_nb = ImO.gyro_orient(R_nb, gyro, dt=dt)
    rpy_gyro = R.from_matrix(R_nb).as_euler('xyz', degrees=True)

    # orientation measurement from accelerometer and magnetometer
    rpy_accel = ImO.accel_mag_orient(filtered_accel, mag=filtered_mag)

    # rpy_accel = None

    # fuse gyro and accel/magnetometer orientation
    kf_orient = ImO.fuse_gyro_accel_mag(np.rad2deg(gyro), rpy_accel, kf_orient)
    # kf_orient = ImO.fuse_gyro_accel_mag(rpy_gyro, rpy_accel, kf_orient)

    # integartion equations
    R_kf = R.from_euler('xyz', kf_orient.x[:3], degrees=True).as_matrix()
    acc_trans = np.matmul(R_kf,filtered_accel) - cfg.settings['gravity']
    
    # compute velocity
    v_xyz = v_xyz + dt*(acc_trans)

    # compute position
    # p_xyz = p_xyz + dt*v_xyz + (acc_trans)*( (dt**2)/2 )
    p_xyz = p_xyz + dt*v_xyz


    # Kalman Filter Steps Below
    z_gps = row[22:24].to_numpy().astype(np.float)
    z_gps = np.hstack((z_gps, 168.024536))

    z_ecef = np.array(pymap3d.geodetic2ecef( *z_gps, ell=None, deg=True))

    # print(z_ecef)

    z_enu = np.array(pymap3d.ecef2enu(*z_ecef, *init_gps))[:2]

    # compute difference between current gps position and previos gps position
    
    pos_diff = abs(z_enu - z_enu_prev)

    
    #  fuse only if absolute of pos_diff is greater than 0
    if (pos_diff.all() != 0 and np.all(pos_diff <= 50)) or idx == 1:
        # print(pos_diff)

        # gps velocity
        v_gps = row[24]

        # print(v_gps)

        # magnitude of gps velocity
        v_norm = v_gps

        # compute the velocity in x and y direction
        v_gps_x = v_norm*np.sin(np.deg2rad(-kf_orient.x[2]))
        v_gps_y = v_norm*np.cos(np.deg2rad(-kf_orient.x[2]))

        # velocity vector
        v_gps = np.array([v_gps_x, v_gps_y])

        # measurement
        y = np.hstack((z_enu, v_gps, acc_trans[:2]))
        
        # change measurement size
        kf.dim_z = 6

        # change H matrix
        kf.H = np.array([[1, 0.0, 0.0, 0.0, 0, 0],
                        [0.0, 1, 0.0, 0.0, 0, 0],
                        [0.0, 0, 1, 0, 0, 0],
                        [0.0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]]) 
        
        # change R matrix
        kf.R = np.diag(np.hstack((cfg.measurement['pos_var'], cfg.measurement['vel_var'], cfg.measurement['accel_var'])))
    else:

        v_gps = row[24]

        # magnitude of gps velocity
        v_norm = v_gps

        # compute the velocity in x and y direction
        v_gps_x = v_norm*np.sin(np.deg2rad(-kf_orient.x[2]))
        v_gps_y = v_norm*np.cos(np.deg2rad(-kf_orient.x[2]))

        # velocity vector
        v_gps = np.array([v_gps_x, v_gps_y])

        # measurement
        y = np.hstack((v_gps, acc_trans[:2]))
        
        # change measuremnt size 
        kf.dim_z = 4

        # change H matrix
        kf.H = np.array([[0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
        
        # change R matrix
        kf.R = np.diag(np.hstack((cfg.measurement['vel_var'], cfg.measurement['accel_var'])))
        

    gps_pos.append(z_enu)

    # print(y)
    kf.predict()
    kf.update(y)

    z_enu_prev = z_enu


    rpy = kf_orient.x[:3]
    rpy_ls.append(rpy)
    rpy_gyro_ls.append(rpy_gyro)
    pos_ls.append(p_xyz)
    rpy_accel_ls.append(rpy_accel)
    kf_pos.append(kf.x[:2])
    vel_ls.append(kf.x[2:4])
    accel_ls.append(kf.x[4:])
    

    # z_enu_prev = z_enu

    t_from_start+=dt
    t_ls.append(t_from_start)
    imu_time_ls.append(imu_time)

print(imu_time)

rpy = np.array(rpy_ls).T
rpy_gyro = np.array(rpy_gyro_ls).T
rpy_accel = np.array(rpy_accel_ls).T

# plot orientation
plt.figure(figsize=(12,9))
plt.plot(rpy[0, :], label='Roll kf')
plt.plot(rpy[1, :], label='Pitch kf')
plt.plot(rpy[2, :], label='Yaw kf')

# plt.plot(rpy_gyro[0, :], label='Roll gyro')
# plt.plot(rpy_gyro[1, :], label='Pitch gyro')
# plt.plot(rpy_gyro[2, :], label='Yaw gyro')

# uncomment this line to view roll pitch yaw from magnetometer as well
# plt.plot(rpy_accel[0, :], label='Roll accel')
# plt.plot(rpy_accel[1, :], label='Pitch accel')
# plt.plot(rpy_accel[2, :], label='Yaw mag')

plt.grid()
plt.legend()
# plt.savefig(filename + '_orientation.png', bbox_inches='tight', dpi=500)

# plot position
pos = np.array(pos_ls).T
# print(pos)
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
# plt.savefig(filename + '_pos.png', bbox_inches='tight', dpi=500)

# plot kf pos and gps pos
kf_track = np.array(kf_pos).T
gps_track = np.array(gps_pos).T

plot_pos(kf_track, gps_track)

# plot velocity
vel = np.array(vel_ls).T
plt.figure(figsize=(12,9))

plt.plot(vel[0,:], label = 'Velocity Along X-axis')
plt.plot(vel[1,:], label = 'Velocity Along Y-axis')
plt.title('Velocity Plot', fontsize = 18, fontweight = 'bold' )
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Magnitude', fontsize = 16)
plt.grid()
plt.legend()

# plot animation
imu_time = np.array(imu_time_ls).T
# plot_animation(imu_time, imu_time, kf_track, gps_track, video_name = filename)

plt.show()




