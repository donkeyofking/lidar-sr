from utils.birdview import *
from utils.removeground import *
from utils.panorama import *
from time import *
import struct
import sys, getopt

import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d

def readpcd(file):
    return o3d.io.read_point_cloud(file)

if __name__ == '__main__':
    #open3d points structure
    pc = np.load("/home/buaaren/carla/ros_env/src/lidarcapture/scripts/lidar64/000000.npy")    #convert o3d structure to numpy narray structure
    print(pc.shape)
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

    img = point_cloud_to_panorama(pc)
    #plt.imshow(img)
    #plt.show()
    points = panorama_to_point_cloud(img)
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
