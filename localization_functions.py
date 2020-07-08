'''
'''

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math









if __name__ == '__main__':

    filename = 'Lums_Cricket_Ground_Square.csv'

    imu_df = pd.read_csv(filename)

    print(imu_df)

    accel_x = imu_df['BMI120_Accelerometer.x'].to_numpy()
    accel_y = imu_df['BMI120_Accelerometer.y'].to_numpy()
    accel_z = imu_df['BMI120_Accelerometer.z'].to_numpy()

    # print(accel_x)

    accel = np.block([
        [accel_x],
        [accel_y],
        [accel_z]
    ])

    print(accel)
    print(np.shape(accel))

    ####### initial orientation ########

    accel_init = np.block([
        [accel_x],
        [accel_y],
        [accel_z]
    ])

    init_orient(accel_init)

    ########### angles from accel and mag #########
    rpy_ls = []

    # for idx, row in imu_df.iterrows():
    for acc in accel.T:
        # print(acc)

        # acc = row[1:4].to_numpy()

        # print(acc)

        # rpy = accel_mag_orient(filtered_accel)
        rpy = accel_mag_orient(acc)

        print(rpy)

        rpy_ls.append(rpy)

    rpy = np.array(rpy_ls).T

    plt.figure(figsize=(12,9))
    plt.plot(rpy[0, :], label='Roll')
    plt.plot(rpy[1, :], label='Pitch')
    plt.plot(rpy[2, :], label='Yaw')
    plt.grid()
    plt.legend()
    plt.show()


