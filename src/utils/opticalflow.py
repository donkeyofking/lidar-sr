from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import cv2
import numpy as np
import sys

sys.path.append("pyflowlib")
from flowapi import *

# ==============================================================================
#                                                                   1_TO_SCALE
# ==============================================================================
def _1_to_scale(a, min, max, dtype=np.float):
    """reverse operation of scale_to_1()
    """
    return (a*(max -min) + min).astype(dtype)


def get_opticalflow(image0,image1):
    # Calculate dense optical flow by Farneback method
    image0 = _1_to_scale(image0, 0, 255, dtype=np.float)
    image1 = _1_to_scale(image1, 0, 255, dtype=np.float)
    return get_opticalflow_nor(image0,image1)


