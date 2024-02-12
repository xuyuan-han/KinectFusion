#pragma once
#include "data_types_GPU.hpp"

namespace GPU{

    FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                    const CameraParameters& camera_params,
                                    const size_t num_levels, const float depth_cutoff,
                                    const int kernel_size, const float color_sigma, const float spatial_sigma);

} // namespace GPU