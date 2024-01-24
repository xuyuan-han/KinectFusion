#pragma once
#include "data_types_CPU.hpp"
// #include "data_types_GPU.hpp"

namespace GPU{

    FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                    const CameraParameters& camera_params,
                                    const size_t num_levels, const float depth_cutoff,
                                    const int kernel_size, const float color_sigma, const float spatial_sigma);

} // namespace GPU

namespace CPU{

FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                            const CameraParameters& camera_params,
                            const size_t num_levels, const float depth_cutoff,
                            const int kernel_size, const float color_sigma, const float spatial_sigma);
void compute_map(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat & vertexMap, cv::Mat & normalMap, const float & depth_cutoff);

} // namespace CPU