import numpy as np


def pointcloud_2_range(points,
                        channels = 64,
                        image_cols = 1024,
                        ang_start_y = 16.6,
                        max_range = 80.0,
                        min_range = 2.0
                        ):
    ang_res_x = 360.0/float(image_cols) 
    ang_res_y = 33.2/float(image_rows_full-1)
    range_image = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
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
        if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
            continue
        range_image[0, rowId[i], colId[i], 0] = thisRange[i]
    # append range image to array
    return range_image



def range_2_pointcloud(image,
                        channels = 64,
                        image_cols = 1024,
                        ang_start_y = 16.6,
                        max_range = 80.0,
                        min_range = 2.0
                        ):
    if len(thisImage.shape) == 3:
        thisImage = thisImage[:,:,0]

    lengthList = thisImage.reshape(image_rows_high*image_cols)
    lengthList[lengthList > max_range] = 0.0
    lengthList[lengthList < min_range] = 0.0

    x = np.sin(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
    y = np.cos(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
    z = np.sin(self.verticalAngle) * lengthList
    
    points = np.column_stack((x,y,z))
    points = np.delete(points, np.where(lengthList==0), axis=0) # comment this line for visualize at the same speed (for video generation)
    # unfinished
    
    # laserCloudOut = pc2.create_cloud(header, self.fields, points)
    # pubHandle.publish(laserCloudOut)