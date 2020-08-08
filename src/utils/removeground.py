import numpy as np
import math
import time

def set_parameterList(count_of_scan, pointsNum_perScan, angle_resolution_z, angle_bottom, ground_scanIndex):
    ret_parameterList = {"count_of_scan":count_of_scan, "pointsNum_perScan":pointsNum_perScan, "angle_resolution_xy":(360/pointsNum_perScan), "angle_resolution_z":angle_resolution_z, "angle_bottom":angle_bottom, "ground_scanIndex":ground_scanIndex}
    return ret_parameterList

def get_parameterList(lidar):
    if lidar == "HDL-64":
        return set_parameterList(64, 2500, 26.9/63, -25.0, 50)
    elif lidar == "VLP-16":
        return set_parameterList(16, 1800, 2.0, 15.1, 7)

def get_countOfScan(parameterList):
    return parameterList.get("count_of_scan", -1)

def get_pointsNumPerScan(parameterList):
    return parameterList.get("pointsNum_perScan", -1)

def get_angleResolutionXY(parameterList):
    return parameterList.get("angle_resolution_xy", -1)

def get_angleResolutionZ(parameterList):
    return parameterList.get("angle_resolution_z", -1)

def get_angleBottom(parameterList):
    return parameterList.get("angle_bottom", -1)

def get_groundScanIndex(parameterList):
    return parameterList.get("ground_scanIndex", -1)

### get lidar parameter
parameterList_HDL64 = get_parameterList("HDL-64")
#parameterList_VLP16 = config.get_parameterList("VLP-16")
### get parameter end

class preProcessor(object):
    '''
    ground remove and cluster ...
    '''
    def __init__(self):
        # lidar parameter
        self.count_of_scan= get_countOfScan(parameterList_HDL64)
        self.pointsNum_perScan = get_pointsNumPerScan(parameterList_HDL64)
        self.angle_bottom = float(get_angleBottom(parameterList_HDL64))
        self.angle_res_z  = float(get_angleResolutionZ(parameterList_HDL64))
        self.angle_res_xy = float(get_angleResolutionXY(parameterList_HDL64))
        self.groundScanIndex  = get_groundScanIndex(parameterList_HDL64)
        # lidar parameter end

        self.sensorMountAngle = 0
        self.start_angle = 0
        self.end_angle   = 0
        self.angle_diff  = 0
        
        self.range_Matrix     = np.full((self.count_of_scan, self.pointsNum_perScan), float("inf"), float)
        self.groundFlag_Matrix= np.zeros([self.count_of_scan, self.pointsNum_perScan], bool)
        self.labelFlag_Matrix = np.zeros([self.count_of_scan, self.pointsNum_perScan], int)
        self.fullPointClouds  = np.zeros([self.count_of_scan*self.pointsNum_perScan, 4], float)
        self.queueIndexX = np.zeros(self.count_of_scan*self.pointsNum_perScan, int)
        self.queueIndexY = np.zeros(self.count_of_scan*self.pointsNum_perScan, int)
        self.allPushedIndexX = np.zeros(self.count_of_scan*self.pointsNum_perScan, int)
        self.allPushedIndexY = np.zeros(self.count_of_scan*self.pointsNum_perScan, int)
        self.labelCount = int(1)
        self.neighborSearchTable = [(-1,0),(0,1),(0,-1),(1,0)]
        self.segmentTheta = 1.0472

    def ground_filter_linefit(self, pointCloudsIn, slope_threshold):
        '''
        input:  pointClouds (points Set [x, y, z])
        '''
        self.rawPointClouds = pointCloudsIn.copy()
        fullPointClouds  = np.zeros([self.count_of_scan*self.pointsNum_perScan, 3], float)
        # calc start and end angle
        points_num  = self.rawPointClouds.shape[0]
        start_point = self.rawPointClouds[0]
        end_point   = self.rawPointClouds[points_num - 1]
        self.start_angle = -math.atan2(start_point[1], start_point[0])
        self.end_angle   = -math.atan2(end_point[1], end_point[0]) + 2*math.pi
        
        self.angle_diff  = self.end_angle - self.start_angle
        if (self.angle_diff) > (3*math.pi):
            self.end_angle -= (2*math.pi)
        elif self.angle_diff < math.pi:
            self.end_angle += (2*math.pi)
        self.angle_diff = self.end_angle - self.start_angle
        
        start = time.clock()
        # point cloud to image
        for i in range(0, points_num):
            this_point = self.rawPointClouds[i]
            verticle_angle = math.atan2(this_point[2], math.sqrt(this_point[0]**2 + this_point[1]**2)) * 180 / math.pi
            rowIndex = int((verticle_angle - self.angle_bottom)/self.angle_res_z)
            if rowIndex < 0 or rowIndex >= self.count_of_scan:
                continue
            
            horizon_angle = math.atan2(this_point[0], this_point[1]) * 180 / math.pi
            colIndex = - int((horizon_angle-90.0) / self.angle_res_xy) + int(self.pointsNum_perScan / 2)
            if colIndex > self.pointsNum_perScan:
                colIndex -= self.pointsNum_perScan
            
            if colIndex < 0 or colIndex >= self.pointsNum_perScan:
                continue
            
            distance = math.sqrt(this_point[0]**2 + this_point[1]**2 + this_point[1]**2)
            self.range_Matrix[int(rowIndex), int(colIndex)] = distance
            index = rowIndex * self.pointsNum_perScan + colIndex
            fullPointClouds[int(index)] = this_point
        # point cloud projection end

        for j in range(0, self.pointsNum_perScan):   # col
            for i in range(0, self.groundScanIndex): # row
                # transfer to one dimension index
                lowerIndex = i * self.pointsNum_perScan + j
                upperIndex = (i+1) * self.pointsNum_perScan + j

                diff_x = fullPointClouds[upperIndex, 0] - fullPointClouds[lowerIndex, 0]
                diff_y = fullPointClouds[upperIndex, 1] - fullPointClouds[lowerIndex, 1]
                diff_z = fullPointClouds[upperIndex, 2] - fullPointClouds[lowerIndex, 2]
                angle = math.atan2(diff_z, math.sqrt(diff_x**2 + diff_y**2)) * 180 / math.pi
                if abs(angle - self.sensorMountAngle) < slope_threshold:
                    self.groundFlag_Matrix[i, j]  = True
                    self.groundFlag_Matrix[i+1,j] = True
                    fullPointClouds[lowerIndex, 2] = -1000  # groundFlag
                    fullPointClouds[upperIndex, 2] = -1000  # groundFlag
        
        index = np.where(fullPointClouds[:,2]>0)
        indices = np.hstack(index)
        ret_nonGroundPoints = np.squeeze(fullPointClouds[indices])
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        print("origin point: ", self.rawPointClouds.shape)
        print("nonGround point: ", ret_nonGroundPoints.shape)
        
        for i in range(0, self.count_of_scan):
            for j in range(0, self.pointsNum_perScan):
                if self.groundFlag_Matrix[i, j] == False or self.range_Matrix[i, j]>10000:
                    self.labelFlag_Matrix[i, j]=-1
        return ret_nonGroundPoints
    
    def ground_filter_heightDiff(self, pointCloudsIn, img_len, img_width, grid_width, ground_height):
        '''
        input:  pointClouds (points Set [x, y, z])
        img_len: field of view len(x axis, forward)
        img_width: field of view width(y axis, left)
        grid_width:
        ground_height: filter threshold
        '''
        self.rawPointClouds = pointCloudsIn.copy()
        minHeight_Matrix = np.full((img_len, img_width),  10000)
        maxHeight_Matrix = np.full((img_len, img_width), -10000)
        
        start = time.clock()
        for i in range(0, self.rawPointClouds.shape[0]):
            this_point = self.rawPointClouds[i]
            # move lidar to center
            row_id = int(this_point[0] / grid_width + img_len / 2)
            col_id = int(this_point[1] / grid_width + img_width / 2)
            if row_id < img_len and col_id < img_width:
                if this_point[2] < minHeight_Matrix[row_id, col_id]:
                    minHeight_Matrix[row_id, col_id] = this_point[2]
                if this_point[2] > maxHeight_Matrix[row_id, col_id]:
                    maxHeight_Matrix[row_id, col_id] = this_point[2]
        height_Matrix = maxHeight_Matrix - minHeight_Matrix
        
        for i in range(0, self.rawPointClouds.shape[0]):
            this_point = self.rawPointClouds[i]
            # move lidar to center
            row_id = int(this_point[0] / grid_width + img_len / 2)
            col_id = int(this_point[1] / grid_width + img_width / 2)
            if row_id < img_len and col_id < img_width:
                if height_Matrix[row_id, col_id] < ground_height:
                    self.rawPointClouds[i, 2] = -1000  # ground flag
        index = np.where(self.rawPointClouds[:,2] > 0)
        indices = np.hstack(index)
        ret_nonGroundPoints = np.squeeze(self.rawPointClouds[indices])
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        print("origin point: ", self.rawPointClouds.shape)
        print("nonGround point: ", ret_nonGroundPoints.shape)
        return ret_nonGroundPoints
        
    def label_components(self, row, col):
        lineCountFlag = np.zeros(self.count_of_scan, bool)
        self.queueIndexX[0] = row
        self.queueIndexY[0] = col
        queueSize = 1
        queueStartInd = 0
        queueEndInd = 1
        self.allPushedIndexX[0] = row
        self.allPushedIndexY[0] = col
        allPushedIndSize = 1

        while queueSize > 0:
            fromIndX = self.queueIndexX[queueStartInd]
            fromIndY = self.queueIndexY[queueStartInd]
            queueSize -= 1
            queueStartInd += 1
            self.labelFlag_Matrix[fromIndX, fromIndY] = self.labelCount

            for searchDir in self.neighborSearchTable:
                thisIndX = fromIndX + searchDir[0]
                thisIndY = fromIndY + searchDir[1]

                if thisIndX<0 or thisIndX>=self.count_of_scan:
                    continue
                if thisIndY<0:
                    thisIndY = self.pointsNum_perScan - 1
                if thisIndY >= self.pointsNum_perScan:
                    thisIndY = 0
                if self.labelFlag_Matrix[thisIndX, thisIndY] != 0:
                    continue

                d1 = max(self.range_Matrix[fromIndX, fromIndY], self.range_Matrix[thisIndX, thisIndY])
                d2 = min(self.range_Matrix[fromIndX, fromIndY], self.range_Matrix[thisIndX, thisIndY])
                if d1 >= 10000:
                    d1 = 10000
                if searchDir[0] == 0:
                    alpha = self.angle_res_xy / 180 * math.pi
                else:
                    alpha = self.angle_res_z / 180 * math.pi

                angle = math.atan2(d2*math.sin(alpha), (d1-d2*math.cos(alpha)))
                if angle > self.segmentTheta:
                    self.queueIndexX[queueEndInd] = thisIndX
                    self.queueIndexY[queueEndInd] = thisIndY
                    queueSize += 1
                    queueEndInd += 1
                    
                    self.labelFlag_Matrix[thisIndX, thisIndY] = self.labelCount
                    lineCountFlag[thisIndX] = True

                    self.allPushedIndexX[allPushedIndSize] = thisIndX
                    self.allPushedIndexY[allPushedIndSize] = thisIndY
                    allPushedIndSize += 1
        
        feasibleSegment = False
        if allPushedIndSize >= 30:
            feasibleSegment = True
        elif allPushedIndSize >= 5:
            lineCount = 0
            for i in range(0, self.count_of_scan):
                if lineCountFlag[i] == True:
                    lineCount += 1
            if lineCount >= 3:
                feasibleSegment = True
        
        if feasibleSegment == True:
            print("+1")
            self.labelCount += 1
        else:
            for i in range(0, allPushedIndSize):
                self.labelFlag_Matrix[self.allPushedIndexX[i], self.allPushedIndexY[i]] = -1000

    def cloud_segmentation(self, segmentTheta):
        self.segmentTheta = segmentTheta
        for i in range(0, self.count_of_scan):
            for j in range(0, self.pointsNum_perScan):
                if self.labelFlag_Matrix[i, j] == 0:
                    self.label_components(i, j)

    def get_rawPointCloud(self):
        print("raw point: ", self.rawPointClouds.shape)
        return self.rawPointClouds.copy()
    
    def get_segmentedPointCloud(self):
        index = []
        count = 0
        print(self.labelCount)
        for i in range(0, self.count_of_scan):
            for j in range(0, self.pointsNum_perScan):
                if self.labelFlag_Matrix[i, j] <= 0 or self.groundFlag_Matrix[i, j] == True:
                    continue
                else:
                    temp_index = int(i*self.count_of_scan+j)
                    self.fullPointClouds[temp_index, 3] = self.labelFlag_Matrix[i, j]
                    index.append(temp_index)
                    count += 1
        print(count)
        indices = np.hstack(index)
        ret_segmentedPointCloud = np.squeeze(self.fullPointClouds[indices])
        print("segmented point: ", ret_segmentedPointCloud.shape)
        return ret_segmentedPointCloud

    def test_printParameterList(self):
        print(self.count_of_scan)
        print(self.pointsNum_perScan)
        print(self.angle_bottom)
        print(self.angle_res_xy)
        print(self.angle_res_z)

def test_printParameterList():
    print(parameterList_HDL64)
