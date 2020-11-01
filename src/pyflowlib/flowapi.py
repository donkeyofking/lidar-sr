from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2

def get_opticalflow_nor(image0,image1):
    
    s = time.time()
    alpha = 0.012
    ratio = 0.75
    minWidth = 10
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1
    # help(pyflow)
    u, v, im2W = pyflow.coarse2fine_flow(
        image0, image1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, image0.shape[0], image0.shape[1], image0.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # create output image
    hsv = np.zeros((64,720,3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

if __name__=='__main__':
    im1 = np.array(Image.open('examples/car1.jpg'))
    im2 = np.array(Image.open('examples/car2.jpg'))
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    get_opticalflow_nor(im1,im2)