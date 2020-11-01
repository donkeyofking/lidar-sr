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



if __name__ == '__main__':

    import PySimpleGUI as sg

    sg.theme('DarkAmber')	# Add a touch of color
    # All the stuff inside your window.
    layout = [  [sg.Text('Some text on Row 1')],
                [sg.Text('Enter something on Row 2'), sg.InputText()],
                [sg.Button('Ok'), sg.Button('Cancel')] ]

    # Create the Window
    window = sg.Window('Window Title', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':	# if user closes window or clicks cancel
            break
        print('You entered ', values[0])

    window.close()