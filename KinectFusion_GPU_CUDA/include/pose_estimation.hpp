#pragma once
#include "data_types_GPU.hpp"

namespace GPU{
    bool pose_estimation(Eigen::Matrix4f& pose,
                            const FrameData& frame_data,
                            const ModelData& model_data,
                            const CameraParameters& cam_params,
                            const int pyramid_height,
                            const float distance_threshold, const float angle_threshold,
                            const std::vector<int>& iterations);
} // namespace GPU