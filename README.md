# KinectFusion

In this project, we aim to reimplement KinectFusion, a well-known RGB-D dense reconstruction system that permits real-time, high-detail mapping and tracking of complex indoor scenes using a simple depth sensor, based on _Newcombe, Richard A., et al._ **KinectFusion: Real-time dense surface mapping and tracking**. The original KinectFusion used a low-cost depth camera, the Kinect sensor, combined with commodity graphics hardware to generate the 3D surface reconstruction of diverse indoor environments under different lighting conditions. We plan to reimplement the original paper using the TUM RGB-D Dataset, and subsequently test the algorithm in our apartment. First, We will fuse all of the camera and depth data in the TUM RGB-D Dataset to build an implicit surface model of the observed scene in real time, then use the iterative closest point (ICP) algorithm to track sensor pose. The surface measurement is integrated into the scene model maintained with a volumetric, truncated signed distance function (TSDF) representation. After that, we apply raycasting to the signed distance function to compute surface prediction and close the loop in the 3D model between mapping and localization.

Our project not only seeks to reimplement the original system but also aims to introduce potentially novel ideas. We plan to execute the KinectFusion method on RGB-D data collected from the LiDAR sensor on an iPhone, attempting to reconstruct our apartment. Additionally, we aim to incorporate semantic segmentation information into the reconstruction process using an appropriate deep-learning model, exploring whether this supplementary information enhances the reconstruction process.

## Requirements

Please check the README in each folder for the detailed requirements.

## Getting Started

```bash
.
├── Data # please download the TUM RGB-D Dataset and put it here
├── iPhoneFusion_GPU_CUDA
├── KinectFusion_CPU_Multi-threading
├── KinectFusion_GPU_CUDA
└── README.md
```

We implement the KinectFusion algorithm in three ways: the CPU version, the GPU version, and the iPhone version.
The CPU version is implemented using multi-threading to speed up the computation.
The GPU version is implemented using CUDA to speed up the computation and enable real-time operation.
The iPhone version is implemented using the iPhone LiDAR sensor with CUDA to reconstruct the room in real time.


**To Use the Segmentation:**

**Implementation Status:**
Segmentation functionality is currently available only in the C++ multithreaded version. We plan to integrate it into the CUDA and iPhone versions in the near future.

**Data Structure:**
- Ensure that your data folder includes a `segmentation` directory containing segmentation maps for each frame.
- Additionally, include a segmentation text file structured similarly to the RGB text file of the TUM RGB-D dataset. Each line should consist of a timestamp and the corresponding segmentation path.

**Activation:**
To activate segmentation, simply uncomment the line `#define USE_CLASSES` in the `data_types.h` file.

**Visual Representation of Data Structure for segmentation:**
```bash
Data Folder
│
├── segmentation
│ ├── frame_1_segmentation_map.png
│ ├── frame_2_segmentation_map.png
│ ├── ...
│ ├── frame_n_segmentation_map.png
│ └── segmentation.txt
│
└── segmentation.txt
```

In this structure:
- The `segmentation` directory contains all segmentation map images.
- Each frame's segmentation map is stored as a PNG image (`frame_1_segmentation_map.png`, `frame_2_segmentation_map.png`, ..., `frame_n_segmentation_map.png`) within the `segmentation` directory.
- The `segmentation.txt` file lists timestamps and the corresponding paths to segmentation maps, similar to the RGB text file of the TUM RGB-D dataset.


Please refer to the README in each folder for the detailed instructions on how to run the code:

[KinectFusion_CPU_Multi-threading README](./KinectFusion_CPU_Multi-threading/README.md)

[KinectFusion_GPU_CUDA README](./KinectFusion_GPU_CUDA/README.md)

[iPhoneFusion_GPU_CUDA README](./iPhoneFusion_GPU_CUDA/README.md)
