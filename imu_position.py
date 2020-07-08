import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

import localization_utils as lu
from KF import KalmanFilter
import config as cfg


def remove_gravity(acc):
    g = np.array([0, 0, 9.81])
    acc = acc - g

    return acc


def integrate_accel(acc, rpy_kf, vel, pos, dt=0.02):

    # convert rpy to rotation matrix
    R_nb = R.from_euler('xyz', rpy_kf, degrees=True).as_matrix()
    # R_nb = lu.eulerAnglesToRotationMatrix(np.deg2rad(rpy_kf))
    # print(rpy_kf)

    # transform accekeration to navigation frame
    acc_trans = np.matmul(R_nb,acc)

    # remove gravity
    acc_trans = remove_gravity(acc_trans)
    # print(acc_trans)

    # compute velocity
    vel = vel + dt*(acc_trans)

    # compute position
    # pos = pos + dt*vel + (acc_trans)*( (dt**2)/2 )
    pos = pos + dt*vel

    return pos, vel


def differentiate_pos(pf, pi, dt):

    v = (pf - pi)/ dt

    return v
