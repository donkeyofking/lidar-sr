from utils.birdview import *
from utils.removeground import *
from utils.panorama import *
from utils.voxel import *
from utils.frontview import *

from time import *
import struct
import sys, getopt

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
                            h_res = 0.3,
                            v_height = (-3, 13),
                            d_range = (0,80)
                            )
    plt.imshow(img)
    plt.show()
    points = panorama_to_point_cloud(img,
                                    v_res = 0.2,
                                    h_res = 0.3,
                                    v_height = (-3, 13),
                                    d_range = (0,80)
                                    )
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


if __name__ == '__main__':
    #open3d points structure
    pc = np.load("/home/buaaren/carla/ros_env/src/lidarcapture/scripts/lidar64/000000.npy")    #convert o3d structure to numpy narray structure
    print(pc.shape)
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    # bev_test(pc)
    # fov_test(pc)
    panorama_test(pc)