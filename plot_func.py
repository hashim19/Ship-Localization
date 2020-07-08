import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_pos(pos_kf, gps_track=None, settings=None, frame='ENU'):

    # get gps data intoa numpy array
    # gps_track = gps_data[['pos_enu_x', 'pos_enu_y']].to_numpy().T

    # plot kf pos and gps track
    #setup limits of x-axis and y axis equal
    if max(pos_kf[0,:]) > max(pos_kf[1,:]):
        right = max(pos_kf[0,:]) + 2
    else:
        right = max(pos_kf[1,:]) + 2
    if min(pos_kf[0,:]) < min(pos_kf[1,:]):
        left = min(pos_kf[0,:]) - 2
    else:
        left = min(pos_kf[1,:]) - 2
    
    plt.figure(figsize=(12,9))
    
    if frame == 'ENU':

        if gps_track is None:
            plt.plot(pos_kf[0,:],pos_kf[1,:], 'g', linewidth=3, label = 'KF pos')
            plt.xlim(left, right)
            plt.ylim(left, right)
        else:

            # kf start and stop point
            plt.plot(pos_kf[0,0], pos_kf[1,0], 'gp', markersize = 8, label = 'Start KF')
            plt.plot(pos_kf[0,len(pos_kf[0,:])-1], pos_kf[1,len(pos_kf[1,:])-1], 'rp', markersize = 8, label = 'Stop KF')

            # gps start and stop point
            plt.plot(gps_track[0,0], gps_track[1,0], 'gP', markersize = 8, label = 'Start Gps')
            plt.plot(gps_track[0,len(gps_track[0,:])-1], gps_track[1,len(gps_track[1,:])-1], 'rP', markersize = 8, label = 'Stop Gps')

            plt.plot(pos_kf[0,:],pos_kf[1,:], 'g', linewidth=3, label = 'KF pos')
            plt.scatter(gps_track[0,:],gps_track[1,:], linewidth=1, label = 'Gps Track')
            plt.xlim(left, right)
            plt.ylim(left, right)
    if frame == 'NED':

        if gps_track is None:
            plt.plot(pos_kf[0,:],pos_kf[1,:], 'g', linewidth=3, label = 'KF pos')

        else:

            plt.plot(pos_kf[1,0], pos_kf[0,0], 'gp', markersize = 8, label = 'Start KF')
            plt.plot(pos_kf[1,len(pos_kf[1,:])-1], pos_kf[0,len(pos_kf[0,:])-1], 'rp', markersize = 8, label = 'Stop KF')

            plt.plot(gps_track[1,0], gps_track[0,0], 'gP', markersize = 8, label = 'Start Gps')
            plt.plot(gps_track[1,len(gps_track[1,:])-1], gps_track[0,len(gps_track[0,:])-1], 'rP', markersize = 8, label = 'Stop Gps')

            plt.plot(pos_kf[1,:],pos_kf[0,:], 'g', linewidth=3, label = 'KF pos')
            # plt.scatter(pos_kf[1,:],pos_kf[0,:], linewidth=1)
            plt.scatter(gps_track[1,:],gps_track[0,:], linewidth=1, label = 'Gps Track')

    plt.title('Position in ENU (Local Frame)', fontsize = 18, fontweight = 'bold' )
    plt.xlabel('East (m)', fontsize = 16)
    plt.ylabel('North (m)', fontsize = 16)
    plt.grid()
    plt.legend()
