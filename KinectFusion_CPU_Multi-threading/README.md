# KinectFusion

In this project, we aim to reimplement KinectFusion, a well-known RGB-D dense reconstruction system that permits real-time, high-detail mapping and tracking of complex indoor scenes using a simple depth sensor, based on _Newcombe, Richard A., et al._ **KinectFusion: Real-time dense surface mapping and tracking**. The original KinectFusion used a low-cost depth camera, the Kinect sensor, combined with commodity graphics hardware to generate the 3D surface reconstruction of diverse indoor environments under different lighting conditions. We plan to reimplement the original paper using the TUM RGB-D Dataset, and subsequently test the algorithm in our apartment. First, We will fuse all of the camera and depth data in the TUM RGB-D Dataset to build an implicit surface model of the observed scene in real-time, then use the iterative closest point (ICP) algorithm to track sensor pose. The surface measurement is integrated into the scene model maintained with a volumetric, truncated signed distance function (TSDF) representation. After that, we apply raycasting to the signed distance function to compute surface prediction and close the loop in the 3D model between mapping and localization.

Our project not only seeks to reimplement the original system but also aims to introduce potentially novel ideas. We plan to execute the KinectFusion method on RGB-D data collected from the LiDAR sensor on an iPhone, attempting to reconstruct our apartment. Additionally, we aim to incorporate semantic segmentation information into the reconstruction process using an appropriate deep-learning model, exploring whether this supplementary information enhances the reconstruction process.

## Requirements for CPU version of KinectFusion without CUDA

We will implement KinectFusion using C++. Essential libraries such as OpenCV for image processing and Eigen for efficient matrix and vector operations will also be utilized.

Regarding the dataset, we will use the TUM RGB-D Dataset, to validate our reconstruction implementation.

## Getting Started

### Prerequisites

- C++14
- OpenCV 4
- Eigen 3

### Download the TUM RGB-D Dataset

```bash
# make sure you are in the *root directory* of this project
cd KinectFusion
mkdir Data && cd Data
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xvzf rgbd_dataset_freiburg1_xyz.tgz
```

### Set the path to your OpenCV and Eigen

```bash
# make sure you are in the *KinectFusion_CPU_Multi-threading* directory under the root directory of this project
cd KinectFusion_CPU_Multi-threading

# create a new cmake configuration file (or you can use any text editor to create this file)
touch config.cmake
echo "set(OpenCV_DIR /path/to/your/opencv)" >> config.cmake # set the path to your OpenCV 4
echo "set(Eigen3_DIR /path/to/your/eigen)" >> config.cmake # set the path to your Eigen 3
```

For example, you can set the path to your OpenCV as follows:

```bash
# config.cmake
SET(OpenCV_DIR "~/Libs/opencv/4.8.0/install/lib/cmake/opencv4")
```

### Building

```bash
# make sure you are in the *KinectFusion_CPU_Multi-threading* directory under the root directory of this project
cd KinectFusion_CPU_Multi-threading
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16
```

### Running

```bash
# make sure you are in the *KinectFusion_CPU_Multi-threading* directory under the root directory of this project
cd KinectFusion_CPU_Multi-threading
mkdir -p output # create a new directory to store the output
cd build
./KinectFusion
```

### Results

The reconstructed 3D model and output videos will be stored in the `output` directory.
