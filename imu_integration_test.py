import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from KF import KalmanFilter


filename = './Data3/Resting_T25-05-2020_17-42-27.csv'
imu_data = pd.read_csv(filename)
# print(np.shape(imu_data))

print(imu_data)

v_t = np.array([0, 0, 0])
p_t = np.array([0, 0, 0])
bias_accel = np.array([0, -0, 9.81])
T =0.05

pos_track = []

# change the numbers to control the number of seconds :1000 means 50 seconds since time difference 0.05 
for idx, row in imu_data[:2000].iterrows():
    accel = row[1:4].to_numpy()
    gyro = row[4:7].to_numpy()
    
    p_t = p_t + T*v_t + (accel - bias_accel)*( (T**2)/2 )
    v_t = v_t + T*(accel - bias_accel)
    
    print(p_t)

    pos_track.append(p_t)


pos = np.array(pos_track).T


# plot distance traveled
plt.figure(figsize=(12,9))

plt.plot(pos[0,:], linewidth=3, label = 'X-Axis')
plt.plot(pos[1,:], linewidth=3, label = 'Y-Axis')
# plt.plot(pos[2,:], linewidth=3, label = 'Z-Axis')

plt.title('IMU Position', fontsize = 18, fontweight = 'bold' )
plt.xlabel('East (m)', fontsize = 16)
plt.ylabel('North (m)', fontsize = 16)
plt.grid()
plt.legend()
plt.show()