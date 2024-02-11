#pragma once
#include "data_types.hpp"

FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                            const CameraParameters& camera_params,
                            const size_t num_levels, const float depth_cutoff,
                            const int kernel_size, const float color_sigma, const float spatial_sigma);
void compute_map(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat & vertexMap, cv::Mat & normalMap, const float & depth_cutoff);

void compute_map_multi_threads(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat& vertexMap, cv::Mat& normalMap, const float& depth_cutoff);

void compute_vertex_map(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat& vertexMap, const float& depth_cutoff, int start_row, int end_row);

void compute_normal_map(const cv::Mat_<float>& depthmap, const cv::Mat& vertexMap, cv::Mat& normalMap, int start_row, int end_row);
