# iPhoneFusion based on GPU version of KinectFusion with CUDA

## Requirements for iPhoneFusion based on GPU version of KinectFusion with CUDA

We will implement KinectFusion using C++ and CUDA (specifically CUDA 11) to meet the real-time computing requirements. Essential libraries such as OpenCV for image processing and Eigen for efficient matrix and vector operations will also be utilized.

In this iPhone version of KinectFusion, we use the [Record3d APP](https://record3d.app/) on iPhone to reconstruct a room in our apartment in real time using the previously mentioned iPhone LiDAR data stream.

## Getting Started

### Prerequisites

- C++17
- CUDA 11
- OpenCV 4
- Eigen 3

- iPhone with LiDAR sensor (iPhone 12 Pro or later)
- [Record3d library](https://github.com/marek-simonik/record3d) (for iPhone LiDAR data)

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

#### Install Record3d library

To use the iPhone LiDAR stream in this project, you need to install the Record3d app on your iPhone and then follow the instructions on the [official Record3d library installation guide](https://github.com/marek-simonik/record3d).

```bash
# clone the Record3d library
cd ~/Libs # or any directory you prefer
git clone https://github.com/marek-simonik/record3d
cd record3d
```

<u>Two modifications to the Record3d library</u> are needed to the Record3d library to make it work with our project. 

```cpp
// Add the following code to the public section of the class Record3DStream in the file `record3d/include/record3d/Record3DStream.h`

/**
 * @brief Get the Current Intrinsic Matrix 
 * 
 * @return std::vector<float> (fx, fy, tx, ty)
 */
std::vector<float> GetCurrentIntrinsicMatrix();
```

```cpp
// Add the following code to the the public part of namespace Record3D in the file `record3d/src/Record3DStream.cpp`
std::vector<float> Record3DStream::GetCurrentIntrinsicMatrix()
{
    std::vector<float> intrinsicMatrix;
    intrinsicMatrix.resize( 4 );
    intrinsicMatrix[ 0 ] = rgbIntrinsicMatrixCoeffs_.fx;
    intrinsicMatrix[ 1 ] = rgbIntrinsicMatrixCoeffs_.fy;
    intrinsicMatrix[ 2 ] = rgbIntrinsicMatrixCoeffs_.tx;
    intrinsicMatrix[ 3 ] = rgbIntrinsicMatrixCoeffs_.ty;

    return intrinsicMatrix;
}
```

Build the Record3d library and install it. Refer to the [official Record3d library installation guide](https://github.com/marek-simonik/record3d).

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 record3d_cpp
make install
```

### Download the TUM RGB-D Dataset

```bash
# make sure you are in the root directory of this project
cd KinectFusion
mkdir Data && cd Data
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xvzf rgbd_dataset_freiburg1_xyz.tgz
```

### Set the path to your OpenCV and Eigen

```bash
# make sure you are in the *iPhoneFusion_GPU_CUDA* directory under the root directory of this project
cd iPhoneFusion_GPU_CUDA

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
# make sure you are in the *iPhoneFusion_GPU_CUDA* directory under the root directory of this project
cd iPhoneFusion_GPU_CUDA
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16
```

### Running

```bash
cd iPhoneFusion_GPU_CUDA/build
./iPhoneFusion_CUDA
```

### Results

The reconstructed 3D model and output videos will be stored in the `output` directory.
