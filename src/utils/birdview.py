import numpy as np
import matplotlib.pyplot as plt
import math


# ==============================================================================
#                                                                   SCALE_TO_1
# ==============================================================================
def scale_to_1(a, min, max, dtype=np.float):
    """ Scales an array of values from specified min, max range to 0-1
        Optionally specify the data type of the output (default is float)
    """
    return (((a - min) / float(max - min)) ).astype(dtype)


# ==============================================================================
#                                                                   1_TO_SCALE
# ==============================================================================
def _1_to_scale(a, min, max, dtype=np.float):
    """reverse operation of scale_to_1()
    """
    return (a*(max -min) + min).astype(dtype)


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-100., 100.),  # left-most to right-most
                           fwd_range = (-100., 100.), # back-most to forward-most
                           height_range=(-3., 5.),  # bottom-most to upper-most
                           ):

    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = -z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-1
    pixel_values = scale_to_1(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values
    # im[y_img, x_img] = 255

    return im




def logpolarize(img , origin=(1000, 1000), base=1.023):
    # base range(1.)
    # ------------> axis-y
    # |
    # |
    # |     .(1000,300)
    # |
    # |
    # |
    # |
    #   axis-x    
    if base<=1 or base>2:
        print("base is not proper, range(1,2)")
        return
    print(img.shape)
    logpolar = np.zeros(img.shape, dtype=np.uint8)
    points  = []
    pixel_values = []
    angle = []
    element = np.nditer(img, flags=['multi_index'])
    while not element.finished:
        if element.value != 0:
            pixel_values.append(element.value)            
            points.append((element.multi_index[0]-origin[0],element.multi_index[1]-origin[1]))
            angle.append(math.atan2(element.multi_index[1]-origin[1],element.multi_index[0]-origin[0]))
        element.iternext()
    points = np.array(points)
    pixel_values = np.array(pixel_values)
    angle = np.array(angle)
    length = np.sqrt( np.square(points[:,0]) + np.square(points[:,1]) )
    def varlog(x):
        para = [-9.15750915750264e-10,8.88278388278302e-06,1.01000000000000]
        return para[0]*x**2 + para[1]*x + para[2] 
    varloggg= np.frompyfunc(varlog,1,1)(length)
    varloggg = varloggg.astype(np.float64)
    length = np.log(length)/np.log(varloggg)
    points[:,0]= np.cos(angle)*length + origin[0]
    points[:,1]= np.sin(angle)*length + origin[1]
    points = points.astype(np.int)
    points[:,0]= np.clip(points[:,0],0,img.shape[0]-1)
    points[:,1]= np.clip(points[:,1],0,img.shape[1]-1)
    print(points.shape)
    logpolar[points[:,0],points[:,1]] = 255
    plt.imshow(logpolar, cmap="gist_heat", vmin=0, vmax=255)
    plt.show()
    return logpolar
    