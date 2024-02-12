# CPU version of KinectFusion without CUDA

## Requirements for CPU version of KinectFusion without CUDA

We will implement KinectFusion using C++. Essential libraries such as OpenCV for image processing and Eigen for efficient matrix and vector operations will also be utilized.

Regarding the dataset, we will use the TUM RGB-D Dataset, to validate our reconstruction implementation.

## Getting Started

### Prerequisites

- C++17
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
cd KinectFusion_CPU_Multi-threading/build
./KinectFusion
```

### Results

The reconstructed 3D model and output videos will be stored in the `output` directory.
