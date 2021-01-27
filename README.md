# A novel TCN-based point cloud super resolution model for mechanical LiDAR


Supplementary material for our ICRA 2021 paper

Abstract - Mechanical LiDAR is an important component of autonomous vehicle perception sensor solutions. In order to enhance the sparse point cloud of low-cost LiDAR scans for outdoor environment perception, this paper presents a novel Temporal Convolutional Network (TCN)-based U-Net model for point cloud super resolution/upsampling. We first project the 3D point cloud onto a 2D image plane, and extend a U-Net convolutional neural network with TCN for solving continues
frames. This helps the model learning the temporal continuity information captured within each single scanning of LiDAR, each time we generate a dense/up-sampled image from last 16 consecutive frames, and project it back to the 3D space as final result. Considering the intrinsic noise of LiDAR, the structural similarity index(SSIM) is introduced as loss function. Experimental studies on both simulator generated datasets and a small scale dataset collected from real road condition with local vehicle platform and show that the proposed model achieves high Peak Signal-to-Noise Ratio (PSNR). Results show that our upsampled point cloud is almost indistinguishable from the real LiDAR point cloud.


# Workflow
![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/workflow.png)

# Use a self-driving chasis to collect data and build its digital twin in carla
![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/platform.png)

# The data collected from the carla
![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/dataset_carla.png)

# The data collected from the real world
![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/dataset_128.png)

# The projecting method
![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/projecting.png)

# Network

# Results
![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/result_carla.png)

![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/result_tunnel.png)

![image](https://github.com/donkeyofking/lidar-sr/blob/master/pics/result_ruby.png)
