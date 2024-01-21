# KinectFusion

In this project, we aim to reimplement KinectFusion, a well-known RGB-D dense reconstruction system that permits real-time, high-detail mapping and tracking of complex indoor scenes using a simple depth sensor, based on _Newcombe, Richard A., et al._ **KinectFusion: Real-time dense surface mapping and tracking**. The original KinectFusion used a low-cost depth camera, the Kinect sensor, combined with commodity graphics hardware to generate the 3D surface reconstruction of diverse indoor environments under different lighting conditions. We plan to reimplement the original paper using the TUM RGB-D Dataset, and subsequently test the algorithm in our apartment. First, We will fuse all of the camera and depth data in the TUM RGB-D Dataset to build an implicit surface model of the observed scene in real-time, then use the iterative closest point (ICP) algorithm to track sensor pose. The surface measurement is integrated into the scene model maintained with a volumetric, truncated signed distance function (TSDF) representation. After that, we apply raycasting to the signed distance function to compute surface prediction and close the loop in the 3D model between mapping and localization.

Our project not only seeks to reimplement the original system but also aims to introduce potentially novel ideas. We plan to execute the KinectFusion method on RGB-D data collected from the LiDAR sensor on an iPhone, attempting to reconstruct our apartment. Additionally, we aim to incorporate semantic segmentation information into the reconstruction process using an appropriate deep-learning model, exploring whether this supplementary information enhances the reconstruction process.

## Requirements

We will implement KinectFusion using C++ and CUDA (specifically CUDA 11) to meet the real-time computing requirements. Essential libraries such as OpenCV for image processing and Eigen for efficient matrix and vector operations will also be utilized.

Regarding the dataset, we will use the TUM RGB-D Dataset, to validate our reconstruction implementation. Additionally, we aim to reconstruct a room in our apartment in real-time using the previously mentioned iPhone LiDAR data stream. Alternatively, we may use a Kinect v1 sensor if one becomes available through the course.

## OpenCV Installation

```Bash
cd ~/Libs/opencv/3.4.16/build &&
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=~/Libs/opencv/3.4.16/install \
-D OPENCV_EXTRA_MODULES_PATH=~/Libs/opencv/contrib-3.4.16/modules \
-D WITH_CUDA=ON \
-D BUILD_DOCS=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
.. &&
make -j16 &&
# cd ./doc/ &&
# make -j16 doxygen &&
make install &&
cd ~/Libs/opencv/4.8.0/build &&
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=~/Libs/opencv/4.8.0/install \
-D OPENCV_EXTRA_MODULES_PATH=~/Libs/opencv/contrib-4.8.0/modules \
-D WITH_CUDA=ON \
-D BUILD_DOCS=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
.. &&
make -j16 &&
make install
```
