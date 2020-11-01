from utils.birdview import *
from utils.removeground import *
from utils.panorama import *
from utils.voxel import *
from utils.frontview import *
from utils.range import *

from time import *
import struct
import sys, getopt
import os

import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d
import open3d
from mpl_toolkits.mplot3d import Axes3D

import cv2

def showVoxel(voxel):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
 
    ax.voxels(voxel, edgecolor="k")
    plt.show()

def readpcd(file):
    return o3d.io.read_point_cloud(file)


def voxel_test(pc):
    pc_voxel = point_cloud_to_voxel(pc)
    print(pc_voxel.shape)
    pc = voxel_to_point_cloud(pc_voxel)
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])


def panorama_test(pc):
    print(pc.shape)
    img = point_cloud_to_panorama(pc,
                            v_res = 0.2,
                            h_res = 0.5,
                            v_height = (-3,3),
                            d_range = (0,80)
                            )
    plt.imshow(img)
    plt.show()
    points = panorama_to_point_cloud(img,
                                    v_res = 0.2,
                                    h_res = 0.5,
                                    v_height = (-3, 3),
                                    d_range = (0,80)
                                    )
    print(points.shape)
    pcd=o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def bev_test(pc):
    img = point_cloud_2_birdseye(pc)
    plt.imshow(img)
    plt.show()

def fov_test(pc):
    lidar_to_2d_front_view(pc)
    # plt.imshow(image)
    # plt.show()

i=0

fig = plt.figure()

def on_key_press(event):
    global i
    if event.key == 'n':
        i +=1
    if event.key == 'p':
        i -=1
    if i<0:
        i=0
        print("At first Image")
    if i>=len(gt64_files):
        i=len(gt64_files)-1
        print("At Last Image")
    print(i)
    pred = np.load(os.path.join(pred64,pred64_files[i]))
    gt = np.load(os.path.join(gt64,gt64_files[i]))
    gt = point_cloud_to_panorama(gt,
                                    v_res = 0.1,# unit meter
                                    h_res = 0.5,# unit degree
                                    v_height = (-3, 3.4)#unit meters
                                    )
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(pred)
    plt.subplot(2,1,2)
    plt.imshow(gt)
    fig.canvas.draw_idle()

if __name__ == '__main__':

    #open3d points structure
    
    # real64 = np.load('/home/buaaren/lidar-sr-dataset/carla64.npy')
    # sample = real64[1030]
    # low_res_index = range(0, 64, 4)

    real128 = np.load('/home/buaaren/lidar-sr-dataset/carla64.npy')
    pred128 = np.load('/home/buaaren/lidar-sr/src/prediction64.npy')
    pc_index = 4000

    pc_real = real128[pc_index+16]
    pc_pred = pred128[pc_index]
    pc_real = np.squeeze(pc_real)
    pc_pred = np.squeeze(pc_pred)
    print(pc_real.shape)
    print(pc_pred.shape)
    low_res_index = range(0, 64, 4)
    print(low_res_index)
    # print(sample.shape)
    # print(sample)
    pc_pred[low_res_index,:] = pc_real[low_res_index,:]
    pc_pred[0:48,:] = pc_real[0:48,:]

    pc_32 = pc_real[low_res_index,:]
    sample = pc_32
    # sample = pc_pred
    # sample = pc_real
    plt.imshow(sample,cmap='plasma')
    #plt.show()

    # sample = range_2_pointcloud(sample,
    #                     channels = 128,
    #                     image_cols = 1808,
    #                     ang_start_y = 25,
    #                     ang_y_total = 40,
    #                     max_range = 160.0,
    #                     min_range = 3.0
    #                     )
    sample = range_2_pointcloud(sample,
                        channels = 16,
                        image_cols = 1024,
                        ang_start_y = 16.6,
                        ang_y_total = 33.2,
                        max_range = 80.0,
                        min_range = 2.0
                        )
    # print(sample.shape)
    # pred = np.load(os.path.join(pred64,pred64_files[i]))
    # gt = np.load(os.path.join(gt64,gt64_files[i]))
    # gt = point_cloud_to_panorama_2(gt,
    #                             v_res = 0.1,# unit meter
    #                             h_res = 0.5,# unit degree
    #                             v_height = (-3, 3.4)#unit meters
    #                             )

    # pred = panorama_to_point_cloud(pred,
    #                             v_res = 0.1,# unit meter
    #                             h_res = 0.5,# unit degree
    #                             v_height = (-3, 3.4)#unit meters
    #                             )

    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)

    # cl, ind = down_pcd.remove_statistical_outlier(20, 2.0)
    # down_pcd = down_pcd.select_down_sample(ind)
    # # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    vis.run()
    vis.destroy_window()

    # bev_test(pc)
    # fov_test(pc)
    # panorama_test(pc)