#pragma once
#include "data_types_GPU.hpp"

namespace GPU{
    void surface_prediction(const VolumeData& volume,
                            cv::cuda::GpuMat& model_vertex,
                            cv::cuda::GpuMat& model_normal,
                            cv::cuda::GpuMat& model_color,
                            const CameraParameters& cam_parameters,
                            const float truncation_distance,
                            const Eigen::Matrix4f& pose);
} // namespace GPU
