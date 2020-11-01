from utils.birdview import *
from utils.removeground import *
from utils.panorama import *
from utils.voxel import *
from utils.frontview import *

from time import *
import struct
import sys, getopt
import os

import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D


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

def panorama_2_test(pc):
    print(pc.shape)
    img = point_cloud_to_panorama_2(pc,
                            v_res = 0.2,
                            h_res = 0.5,
                            v_height = (-3,3),
                            d_range = (0,80)
                            )
    plt.imshow(img[:,:,0])
    plt.show()
    points = panorama_to_point_cloud_2(img,
                                    v_res = 0.2,
                                    h_res = 0.5,
                                    v_height = (-3, 3),
                                    d_range = (0,80),
                                    threshold = 0.9
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
gt64 = "/home/buaaren/lidar-sr-dataset/gt64"
pred64 = "/home/buaaren/lidar-sr-dataset/pred64"
gt64_files = sorted(os.listdir(gt64))
pred64_files = sorted(os.listdir(pred64))
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
    pred_image = pred[:,:,0]
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(pred_image)
    plt.subplot(2,1,2)
    plt.imshow(gt)
    fig.canvas.draw_idle()

if __name__ == '__main__':

    #open3d points structure
    

    pred = np.load(os.path.join(pred64,pred64_files[i]))
    gt = np.load(os.path.join(gt64,gt64_files[i]))
    gt = point_cloud_to_panorama(gt,
                                v_res = 0.1,# unit meter
                                h_res = 0.5,# unit degree
                                v_height = (-3, 3.4)#unit meters
                                )

    print(gt.shape)
    pred_image = pred[:,:,0]
    plt.subplot(2,1,1)
    plt.imshow(pred_image)
    plt.subplot(2,1,2)
    plt.imshow(gt)
    # fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    # fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()
    pred = panorama_to_point_cloud_2(pred,
                                v_res = 0.1,# unit meter
                                h_res = 0.5,# unit degree
                                v_height = (-3, 3.4),#unit meters
                                threshold = 1.01
                                )
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred)
    o3d.visualization.draw_geometries([pcd])
    
    # bev_test(pc)
    # fov_test(pc)
    # panorama_test(gt)   
    # panorama_2_test(gt)