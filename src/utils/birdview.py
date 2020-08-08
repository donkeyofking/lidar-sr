import numpy as np
import matplotlib.pyplot as plt
import math


# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-100., 100.),  # left-most to right-most
                           fwd_range = (-100., 100.), # back-most to forward-most
                           height_range=(-2., 5.),  # bottom-most to upper-most
                           show_flag=False,
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
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
    z_points = z_points[indices]

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

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    #im[y_img, x_img] = pixel_values
    im[y_img, x_img] = 255

    if(show_flag):
        plt.imshow(im, cmap="gist_heat", vmin=0, vmax=255)
        plt.show()
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
    