#pragma once

#include "data_types_GPU.hpp"

namespace GPU{
	void surface_reconstruction(const cv::cuda::GpuMat& depth_image,
								const cv::cuda::GpuMat& color_image,
								VolumeData& volume,
								const CameraParameters& cam_params,
								const float truncation_distance,
								const Eigen::Matrix4f& model_view);
} // namespace GPU