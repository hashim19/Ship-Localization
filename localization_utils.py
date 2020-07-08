# -*- coding: utf-8 -*-

import numpy as np
import math


def Rx(theta):
    R = np.zeros((3,3))
    R[0,0] = 1 
    R[0,1] = 0
    R[0,2] = 0
    R[1,0] = 0
    R[1,1] = math.cos(theta)
    R[1,2] = -math.sin(theta)
    R[2,0] = 0
    R[2,1] = math.sin(theta)
    R[2,2] = math.cos(theta)
    return R


def Ry(theta):
    R = np.zeros((3,3))
    R[0,0] = math.cos(theta) 
    R[0,1] = 0
    R[0,2] = math.sin(theta)
    R[1,0] = 0
    R[1,1] = 1
    R[1,2] = 0
    R[2,0] = -math.sin(theta)
    R[2,1] = 0
    R[2,2] = math.cos(theta)
    return R

def Rz(theta):
    R = np.zeros((3,3))
    R[0,0] = math.cos(theta) 
    R[0,1] = -math.sin(theta)
    R[0,2] = 0
    R[1,0] = math.sin(theta)
    R[1,1] = math.cos(theta)
    R[1,2] = 0
    R[2,0] = 0
    R[2,1] = 0
    R[2,2] = 1
    return R

def Rz_2d(theta):
    R = np.zeros((2,2))
    R[0,0] = math.cos(theta) 
    R[0,1] = -math.sin(theta)
    R[1,0] = math.sin(theta)
    R[1,1] = math.cos(theta)
    return R

def SkewMat(vec):
    sm = np.zeros((3,3))
    sm [0,0] = 0
    sm [0,1] = -vec[2,0]
    sm [0,2] = vec[1,0]
    sm [1,0] = vec[2,0]
    sm [1,1] = 0
    sm [1,2] = -vec[0,0]
    sm [2,0] = -vec[1,0]
    sm [2,1] = vec[0,0]
    sm [2,2] = 0
    return sm

def reshape_z(z, dim_z, ndim):
    """ ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


def normalize_angle(x):
    x = x % (2 * 180)    # force in range [0, 2 pi)
    if x > 180:          # move to [-pi, pi)
        x -= 2 * 180
    return x