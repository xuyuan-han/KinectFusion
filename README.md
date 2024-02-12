# KinectFusion

In this project, we aim to reimplement KinectFusion, a well-known RGB-D dense reconstruction system that permits real-time, high-detail mapping and tracking of complex indoor scenes using a simple depth sensor, based on _Newcombe, Richard A., et al._ **KinectFusion: Real-time dense surface mapping and tracking**. The original KinectFusion used a low-cost depth camera, the Kinect sensor, combined with commodity graphics hardware to generate the 3D surface reconstruction of diverse indoor environments under different lighting conditions. 

We detail our efforts in reimplementing KinectFusion, exploring both multi-threaded CPU and CUDA-based GPU approaches to achieve efficient real-time performance. Our implementation extends the conventional KinectFusion pipeline by incorporating modern advancements such as high-resolution LiDAR data from iPhone sensors and integrating semantic information from 2D segmentation maps to enhance the reconstruction process. Through experiments conducted on the TUM RGB-D dataset and real-world data captured by an iPhone LiDAR sensor, we demonstrate the efficacy of our approach in producing detailed and accurate 3D reconstructions. Our results highlight the significant speedup achieved by the CUDA implementation, achieving real-time reconstruction at 30 fps, which is 14 times faster than its multi-threaded CPU counterpart. This work not only proves that consumer-grade sensors can effectively reconstruct 3D scenes but also highlights how integrating semantic segmentation can enhance reconstruction quality.

Check out our [final report](./Final_Report.pdf) for more details.

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
