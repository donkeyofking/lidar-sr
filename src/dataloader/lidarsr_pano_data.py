#!/usr/bin/env python
import os
import time
#import _thread
import math
import cv2
import numpy as np
import sys
import random
sys.path.append("..")
from utils.panorama import *

import matplotlib.pyplot as plt
import open3d as o3d

path16 = "/home/buaaren/lidar-sr-dataset/lidar16"
path64 = "/home/buaaren/lidar-sr-dataset/lidar64"
test16 = "/home/buaaren/lidar-sr-dataset/lidar16_test"
pred64 = "/home/buaaren/lidar-sr-dataset/pred64"
lidar16_files = sorted(os.listdir(path16))
lidar64_files = sorted(os.listdir(path64))
lidar16_test_files = sorted(os.listdir(test16))

sensor_noise = 0.03
max_range = 80.0
min_range = 2.0
normalize_ratio = 1.0
v_res = 0.1# unit meter
h_res = 0.5# unit degree
v_height = (-3, 3.4)#unit meters
v_max = int((v_height[1] - v_height[0])/v_res)
h_max = int(360/h_res)
channel = 2 #add confidence channel
upscaling_factor = 1



def preprocessdata(npyfile):
    noise = np.random.normal(0, sensor_noise, npyfile.shape)
    noise[npyfile == 0] = 0
    npyfile = npyfile + noise
    # normalize data
    npyfile = npyfile / normalize_ratio
    return npyfile


def generate_data_from_path(batch_size=4):

    count16 = len(lidar16_files)
    count64 = len(lidar64_files)
    if count16 != count64:
        print("training data and label numbers are not equal")
        return        
    print('----------------------------')
    batch_counts = count64 // batch_size
    while 1:
        i = random.randrange(batch_counts - 1)
        index_start =  i    *  batch_size
        index_end   = (i+1) *  batch_size
        # print("read lidar16 data" + str(lidar16_files[index_start:index_end]))
        # print("read lidar64 data" + str(lidar64_files[index_start:index_end]))
        x = []
        for file in lidar16_files[index_start:index_end]:
            x.append(point_cloud_to_panorama_2(preprocessdata(np.load(os.path.join(path16,file))),
                                            v_res = v_res,
                                            h_res = h_res,
                                            v_height = v_height,
                                            d_range = (0,80)
                                            )
                    )
        x = np.asarray(x)
        # plt.imshow(x[0])
        # plt.show()
        # pcd=o3d.geometry.PointCloud()
        # points = panorama_to_point_cloud(x[0])
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])
        # x = x[:, : ,:, np.newaxis]
        # print(x.shape)
        y = []
        for file in lidar64_files[index_start:index_end]:
            y.append(point_cloud_to_panorama_2(preprocessdata(np.load(os.path.join(path64,file))),
                                            v_res = v_res,
                                            h_res = h_res,
                                            v_height = v_height,
                                            d_range = (0,80)
                                            )
                    )
            # y.append(np.load(os.path.join(path64,file)))
        y = np.asarray(y)
        # plt.imshow(y[0])
        # plt.show()
        # y = y[:, : ,:, np.newaxis]
        # pcd=o3d.geometry.PointCloud()
        # points = panorama_to_point_cloud(y[0])
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])
        # print(y.shape)
        # print("i:"+str(i))
        yield (x,y)


def generate_seq_data_from_path(batch_size=4, length=3):
    count16 = len(lidar16_files)
    count64 = len(lidar64_files)
    if count16 != count64:
        print("training data and label numbers are not equal")
        return        
    print('----------------------------')
    batch_counts = count64 // (batch_size * length)
    while 1:
        i = random.randrange(batch_counts - 1)
        index_start =  i    *  batch_size * length
        index_end   = (i+1) *  batch_size * length
        print("")
        print("read lidar16 data" + str(lidar16_files[index_start:index_end]))
        print("read lidar64 data" + str(lidar64_files[index_start:index_end]))
        x = []
        for bs in range(batch_size):
            basket = []
            for leng in range(length):
                index = bs * length + leng + index_start
                file = lidar16_files[index]
                basket.append(point_cloud_to_panorama(preprocessdata(np.load(os.path.join(path16,file))),
                                            v_res = v_res,
                                            h_res = h_res,
                                            v_height = v_height,
                                            d_range = (0,80)
                                            )
                    )
            basket = np.asarray(basket)
            x.append(basket)
        x = np.asarray(x)
        # plt.imshow(x[0])
        # plt.show()
        # pcd=o3d.geometry.PointCloud()
        # points = panorama_to_point_cloud(x[0])
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])
        x = x[:, : ,:, :, np.newaxis]
        print(x.shape)
        y = []
        for bs in range(batch_size):
            basket = []
            for leng in range(length):
                index = bs * length + leng + index_start
                file = lidar16_files[index]
                basket.append(point_cloud_to_panorama(preprocessdata(np.load(os.path.join(path64,file))),
                                            v_res = v_res,
                                            h_res = h_res,
                                            v_height = v_height,
                                            d_range = (0,80)
                                            )
                    )
            basket = np.asarray(basket)
            y.append(basket)
        y = np.asarray(y)
        # # plt.imshow(y[0])
        # # plt.show()
        # y = y[:, : ,:, np.newaxis]
        # # pcd=o3d.geometry.PointCloud()
        # # points = panorama_to_point_cloud(y[0])
        # # pcd.points = o3d.utility.Vector3dVector(points)
        # # o3d.visualization.draw_geometries([pcd])
        # # print(y.shape)
        # # print("i:"+str(i))
        yield (x,y)

if __name__=='__main__':
    # lidar16_files = sorted(os.listdir("/home/buaaren/carla/ros_env/src/lidarcapture/scripts/lidar16"))
    # path = "/home/buaaren/carla/ros_env/src/lidarcapture/scripts/lidar16"
    # for file in lidar16_files:
    #     npy = np.load(os.path.join(path,file))
    #     img = point_cloud_to_panorama(npy)
    #     plt.imshow(img)
    #     plt.show()
    #     print(npy.shape)
    generate_data_from_path(batch_size=3)
    # generate_seq_data_from_path(batch_size=4)



