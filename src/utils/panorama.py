import numpy as np

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
#                                                        POINT_CLOUD_TO_PANORAMA
# ==============================================================================
def point_cloud_to_panorama(points,
                            v_res = 0.1,
                            h_res = 0.3,
                            v_height = (-3, 13),
                            d_range = (0,80)
                            ):
    # print(points.shape)
    # up_points = points[points[:,2]>=0]
    # print(up_points.shape)
    # down_points = points[points[:,2]<0]
    # print(down_points.shape)
    ######## attention : z negitivate means height direction
    points = points[ points[:,2] < -v_height[0] ]
    points = points[ points[:,2] > -v_height[1] ]
    # print(points.shape)
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = -points[:, 2] - v_height[0]
    #r_points = points[:, 3]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    #d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2) # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_height_total = -v_height[0] + v_height[1]

    # CONVERT TO RADIANS
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = z_points / v_res
    y_max = int(np.ceil(v_height_total / v_res))
    y_img = np.trunc(y_img).astype(np.int32)

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = 180.0 / h_res 
    x_img = np.trunc(x_img + x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])
    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max  , x_max ], dtype=np.float)
    img[y_img, x_img] = scale_to_1(d_points, min=d_range[0], max=d_range[1],dtype=np.float)
    # print(img.shape)
    return img


# ==============================================================================
#                                                        PANORAMA_TO_POINT_CLOUD
# ==============================================================================
def panorama_to_point_cloud(image,
                            v_res = 0.1,
                            h_res = 0.3,
                            v_height = (-3, 13),
                            d_range = (0,80)
                            ):


    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_height_total = -v_height[0] + v_height[1]

    # CONVERT TO RADIANS
    h_res_rad = h_res * (np.pi / 180)

    points = []
    pixel_values = []
    element = np.nditer(image, flags=['multi_index'])
    while not element.finished:
        if element.value != 0:
            pixel_values.append(element.value)
            #print(element.multi_index)
            #print(element.value)
            points.append((element.multi_index[0],element.multi_index[1]))
        element.iternext()
    points = np.array(points)
    x_image = points[:,1]
    y_image = points[:,0]
    pixel_values = np.array(pixel_values)
    d_points = _1_to_scale(pixel_values, d_range[0], d_range[1], dtype=np.float)
    z_points = y_image * v_res - v_height[0]
    x_image = x_image - 180.0 / h_res
    y_points = np.sin(x_image * h_res_rad) * d_points
    x_points = np.cos(x_image * h_res_rad) * d_points
    points = np.array([x_points, y_points, z_points])
    points = points.T
    # print(points.shape)
    return points



# ==============================================================================
#                                                        POINT_CLOUD_TO_PANORAMA
# ==============================================================================
def point_cloud_to_panorama_2(points,
                            v_res = 0.1,
                            h_res = 0.3,
                            v_height = (-3, 13),
                            d_range = (0,80)
                            ):
    # print(points.shape)
    # up_points = points[points[:,2]>=0]
    # print(up_points.shape)
    # down_points = points[points[:,2]<0]
    # print(down_points.shape)
    ######## attention : z negitivate means height direction
    points = points[ points[:,2] < -v_height[0] ]
    points = points[ points[:,2] > -v_height[1] ]
    # print(points.shape)
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = -points[:, 2] - v_height[0]
    #r_points = points[:, 3]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    #d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2) # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_height_total = -v_height[0] + v_height[1]

    # CONVERT TO RADIANS
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = z_points / v_res
    y_max = int(np.ceil(v_height_total / v_res))
    y_img = np.trunc(y_img).astype(np.int32)

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = 180.0 / h_res 
    x_img = np.trunc(x_img + x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    # CLIP DISTANCES
    d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])
    # CONVERT TO IMAGE ARRAY
    img = np.zeros([y_max  , x_max ], dtype=np.float)
    img[y_img, x_img] = scale_to_1(d_points, min=d_range[0], max=d_range[1],dtype=np.float)
    confidence = np.zeros([y_max  , x_max ], dtype=np.float)
    confidence[y_img, x_img] = 1.0
    img2c = np.stack((img, confidence), axis=2)
    # print(img.shape)
    # print(confidence.shape)
    # print(img2c.shape)
    return img2c


# ==============================================================================
#                                                        PANORAMA_TO_POINT_CLOUD
# ==============================================================================
def panorama_to_point_cloud_2(image,
                            v_res = 0.1,
                            h_res = 0.3,
                            v_height = (-3, 13),
                            d_range = (0,80),
                            threshold = 0.9
                            ):
    # print(image.shape)
    distance = image[:,:,0]
    confidence = image[:,:,1]
    # print(confidence)
    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_height_total = -v_height[0] + v_height[1]

    # CONVERT TO RADIANS
    h_res_rad = h_res * (np.pi / 180)

    points = []
    pixel_values = []
    # element = np.nditer(distance, flags=['multi_index'])
    filter = np.nditer(confidence, flags=['multi_index'])

    while not filter.finished:
        if filter.value>threshold:
            pixel_values.append(distance[filter.multi_index[0],filter.multi_index[1]])
            #print(element.multi_index)
            #print(element.value)
            points.append((filter.multi_index[0],filter.multi_index[1]))
        filter.iternext()
    if len(points) == 0:
        print("no points at all")
        return
    points = np.array(points)

    x_image = points[:,1]
    y_image = points[:,0]
    pixel_values = np.array(pixel_values)
    d_points = _1_to_scale(pixel_values, d_range[0], d_range[1], dtype=np.float)
    z_points = y_image * v_res - v_height[0]
    x_image = x_image - 180.0 / h_res
    y_points = np.sin(x_image * h_res_rad) * d_points
    x_points = np.cos(x_image * h_res_rad) * d_points
    points = np.array([x_points, y_points, z_points])
    points = points.T
    print(points.shape)
    return points