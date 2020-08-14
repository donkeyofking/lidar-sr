import numpy as np

# ==============================================================================
#                                                                   SCALE_TO_1
# ==============================================================================
def scale_to_1(a, min, max, dtype=np.float):
    """ Scales an array of values from specified min, max range to 0-1
        Optionally specify the data type of the output (default is float)
    """
    return ((a - min) / float(max - min)).astype(dtype)

# ==============================================================================
#                                                                   1_TO_SCALE
# ==============================================================================
def _1_to_scale(a, min, max, dtype=np.float):
    """reverse operation of scale_to_1()
    """
    return (a*(max -min)/min).astype(dtype)



# ==============================================================================
#                                                        POINT_CLOUD_TO_VOXEL   
# ==============================================================================
def point_cloud_to_voxel(points,
                            x_res = 1,
                            y_res = 1,
                            z_res = 1,
                            x_height = (-50, 50),
                            y_height = (-80, 80),
                            z_height = (-3, 13),
                            d_range = (0.1,80)
                            ):
    print("original point number {}".format(points.shape))
    points = points[ points[:,0] >  x_height[0] ]
    points = points[ points[:,0] <  x_height[1] ]
    points = points[ points[:,1] >  y_height[0] ]
    points = points[ points[:,1] <  y_height[1] ]
    points = points[ points[:,2] < -z_height[0] ]
    points = points[ points[:,2] > -z_height[1] ]
    print("afterslice point number {}".format(points.shape))
    # seperate axis 
    x_points = points[:, 0] - x_height[0]
    y_points = points[:, 1] - y_height[0]
    z_points = -points[:, 2] - z_height[0]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2 + points[:,2]**2 )  # map distance relative to origin
    # calculate max range unit in meters
    x_height_total = -x_height[0] + x_height[1]
    y_height_total = -y_height[0] + y_height[1]
    z_height_total = -z_height[0] + z_height[1]
    # max pixel length in every axis 
    x_max = int(np.ceil(x_height_total / x_res))
    y_max = int(np.ceil(y_height_total / y_res))
    z_max = int(np.ceil(z_height_total / z_res))
    # compute img in every axis
    x_img = x_points / x_res
    x_img = np.trunc(x_img).astype(np.int32)
    y_img = y_points / y_res
    y_img = np.trunc(y_img).astype(np.int32)
    z_img = z_points / z_res
    z_img = np.trunc(z_img).astype(np.int32)
    # CONVERT TO IMAGE ARRAY
    img = np.zeros([z_max, y_max , x_max ], dtype=np.float)
    print(img.shape)
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])
#    img[z_img , y_img, x_img] = scale_to_1(d_points, min=d_range[0], max=d_range[1],dtype=np.float)
    img[z_img , y_img, x_img] = 1

    # print(img.shape)
    return img


# ==============================================================================
#                                                        VOXEL_TO_POINT_CLOUD   
# ==============================================================================
def voxel_to_point_cloud(image,
                        x_res = 1,
                        y_res = 1,
                        z_res = 1,
                        x_height = (-50, 50),
                        y_height = (-80, 80),
                        z_height = (-3, 13),
                        d_range = (0.1,80)
                        ):
    # RESOLUTION AND FIELD OF VIEW SETTINGS
    x_height_total = -x_height[0] + x_height[1]
    y_height_total = -y_height[0] + y_height[1]
    z_height_total = -z_height[0] + z_height[1]

    points = []
    pixel_values = []
    element = np.nditer(image, flags=['multi_index'])
    while not element.finished:
        if element.value != 0:
            pixel_values.append(element.value)
            #print(element.multi_index)
            #print(element.value)
            points.append((element.multi_index[0],element.multi_index[1],element.multi_index[2]))
        element.iternext()
    points = np.array(points)
    z_image = points[:,0]
    y_image = points[:,1]
    x_image = points[:,2]
    print(np.max(z_image))
    print(np.max(y_image))
    print(np.max(x_image))

    pixel_values = np.array(pixel_values)
    d_points = _1_to_scale(pixel_values, d_range[0], d_range[1], dtype=np.float)
    z_points = z_image * z_res + z_height[0]
    y_points = y_image * y_res + y_height[0]
    x_points = x_image * x_res + x_height[0]
    print(np.max(z_points))
    print(np.max(y_points))
    print(np.max(x_points))
    points = np.array([x_points, y_points, z_points])
    points = points.T
    print(points.shape)
    return points