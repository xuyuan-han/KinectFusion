# KinectFusion

In this project, we aim to reimplement KinectFusion, a well-known RGB-D dense reconstruction system that permits real-time, high-detail mapping and tracking of complex indoor scenes using a simple depth sensor, based on _Newcombe, Richard A., et al._ **KinectFusion: Real-time dense surface mapping and tracking**. The original KinectFusion used a low-cost depth camera, the Kinect sensor, combined with commodity graphics hardware to generate the 3D surface reconstruction of diverse indoor environments under different lighting conditions. We plan to reimplement the original paper using the TUM RGB-D Dataset, and subsequently test the algorithm in our apartment. First, We will fuse all of the camera and depth data in the TUM RGB-D Dataset to build an implicit surface model of the observed scene in real time, then use the iterative closest point (ICP) algorithm to track sensor pose. The surface measurement is integrated into the scene model maintained with a volumetric, truncated signed distance function (TSDF) representation. After that, we apply raycasting to the signed distance function to compute surface prediction and close the loop in the 3D model between mapping and localization.

Our project not only seeks to reimplement the original system but also aims to introduce potentially novel ideas. We plan to execute the KinectFusion method on RGB-D data collected from the LiDAR sensor on an iPhone, attempting to reconstruct our apartment. Additionally, we aim to incorporate semantic segmentation information into the reconstruction process using an appropriate deep-learning model, exploring whether this supplementary information enhances the reconstruction process.

## Requirements for CPU version of KinectFusion without CUDA

We will implement KinectFusion using C++ and CUDA (specifically CUDA 11) to meet the real-time computing requirements. Essential libraries such as OpenCV for image processing and Eigen for efficient matrix and vector operations will also be utilized.

Regarding the dataset, we will use the TUM RGB-D Dataset, to validate our reconstruction implementation.

## Getting Started

### Prerequisites

- C++17
- CUDA 11
- OpenCV 4
- Eigen 3

#### Install CUDA 11

Please refer to the [NVIDIA website](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local).

#### Install OpenCV 4 with CUDA support

This is a guide to installing OpenCV 4 with CUDA support. Here it is assumed that you have already installed CUDA 11 and downloaded the OpenCV source code to `~/Libs/opencv/4.8.0` and the OpenCV contrib source code to `~/Libs/opencv/contrib-4.8.0`. Please refer to the [official OpenCV installation guide](https://docs.opencv.org/4.8.0/d7/d9f/tutorial_linux_install.html).

```bash
# change the directory to the directory where you downloaded the OpenCV source code
cd ~/Libs/opencv/4.8.0 
mkdir build && cd build
# change the install directory to your preferred directory and set the path to the OpenCV contrib source code
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=~/Libs/opencv/4.8.0/install \
-D OPENCV_EXTRA_MODULES_PATH=~/Libs/opencv/contrib-4.8.0/modules \
-D WITH_CUDA=ON \
-D BUILD_DOCS=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
..
make -j16
make install
```

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
# make sure you are in the *KinectFusion_GPU_CUDA* directory under the root directory of this project
cd KinectFusion_GPU_CUDA

# create a new cmake configuration file (or you can use any text editor to create this file)
touch config.cmake
echo "set(OpenCV_DIR /path/to/your/opencv)" >> config.cmake # set the path to your OpenCV 4
echo "set(Eigen3_DIR /path/to/your/eigen)" >> config.cmake # set the path to your Eigen 3

# set your GPU Compute Capability, refer to https://developer.nvidia.com/cuda-gpus
# for example, if you have a GTX 1080, the compute capability is 6.1. You can set it as follows:
echo "set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_61,code=sm_61 --expt-relaxed-constexpr)" >> config.cmake 
```

For example, you can set the path to your OpenCV as follows:

```bash
# config.cmake
SET(OpenCV_DIR "~/Libs/opencv/4.8.0/install/lib/cmake/opencv4")
```

### Building

```bash
# make sure you are in the *KinectFusion_GPU_CUDA* directory under the root directory of this project
cd KinectFusion_GPU_CUDA
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16
```

### Running

```bash
cd KinectFusion_GPU_CUDA/build
./KinectFusion_CUDA
```

### Results

The reconstructed 3D model and output videos will be stored in the `output` directory.
