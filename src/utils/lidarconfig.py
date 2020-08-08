
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