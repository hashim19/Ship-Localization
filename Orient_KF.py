# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:36:36 2020

@author: zeesh
"""
import numpy as np
import pandas as pd
from copy import deepcopy

import localization_utils as lu


class KalmanFilter:
    
    global x
    
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        
        self.x = np.zeros((dim_x, 1))        # state
        self.P = np.eye(dim_x)               # uncertainty covariance
        self.Q = np.eye(dim_x)               # process uncertainty
        self.B = None                     # control transition matrix
        self.A = np.eye(dim_x)               # state transition matrix
        self.H = np.zeros((dim_z, dim_x))    # Measurement function
        self.R = np.eye(dim_z)               # state uncertainty
        self.z = np.array([[None]*self.dim_z]).T
        
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = np.zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

         # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)
        
        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
        self.inv = np.linalg.inv
        
    def predict(self, u=None, B=None, A=None, Q=None):

        if B is None:
            B = self.B
        if A is None:
            A = self.A
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # print(np.shape(self.x))
        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = np.dot(A, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(A, self.x)

        # print(np.shape(self.x))

        # P = FPF' + Q
        self.P = np.dot(np.dot(A, self.P), A.T) + Q

        
        # self.x[:2] = self.x[:2]*0.97

        self.x[0] = lu.normalize_angle(self.x[0])
        self.x[1] = lu.normalize_angle(self.x[1])
        self.x[2] = lu.normalize_angle(self.x[2])

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
        
    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """


        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        z = lu.reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x)

        # common subexpression for speed
        PHT = np.dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
