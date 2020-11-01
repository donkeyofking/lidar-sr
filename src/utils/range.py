import numpy as np


def pointcloud_2_range(points,
                        channels = 400,
                        image_cols = 1808,
                        ang_start_y = 25,
                        ang_y_total = 40,
                        max_range = 160,
                        min_range = 3.0
                        ):
    ang_res_x = 360.0/float(image_cols) 
    ang_res_y = ang_y_total/float(channels-1)
    range_image = np.zeros((1, channels, image_cols, 1), dtype=np.float32)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    # find row id
    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
    # find column id
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2
    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    # filter range
    thisRange = np.sqrt(x * x + y * y + z * z)
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0
    # save range info to range image
    for i in range(len(thisRange)):
        if rowId[i] < 0 or rowId[i] >= channels or colId[i] < 0 or colId[i] >= image_cols:
            continue
        range_image[0, rowId[i], colId[i], 0] = thisRange[i]
    # append range image to array
    return range_image



def range_2_pointcloud(thisImage,
                        channels = 128,
                        image_cols = 1808,
                        ang_start_y = 25,
                        ang_y_total = 40,
                        max_range = 160.0,
                        min_range = 3.0
                        ):


    if len(thisImage.shape) == 3:
        thisImage = thisImage[:,:,0]
    lengthList = thisImage.reshape(channels*image_cols)
    lengthList[lengthList > max_range] = 0.0
    lengthList[lengthList < min_range] = 0.0

    rowList = []
    colList = []
    for i in range(channels):
        rowList = np.append(rowList, np.ones(image_cols)*i)
        colList = np.append(colList, np.arange(image_cols))

    ang_res_x = 360.0/float(image_cols) 
    ang_res_y = ang_y_total/float(channels-1)

    verticalAngle = np.float32(rowList * ang_res_y) - ang_start_y
    horizonAngle = - np.float32(colList + 1 - (image_cols/2)) * ang_res_x + 90.0
    
    verticalAngle = verticalAngle / 180.0 * np.pi
    horizonAngle = horizonAngle / 180.0 * np.pi

    x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList
    y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList
    z = np.sin(verticalAngle) * lengthList
    
    points = np.column_stack((x,y,z))
    points = np.delete(points, np.where(lengthList==0), axis=0) # comment this line for visualize at the same speed (for video generation)

    return points
