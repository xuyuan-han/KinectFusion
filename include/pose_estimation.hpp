#pragma once
#include "data_types.hpp"

bool pose_estimation(Eigen::Matrix4f& pose,
                    const FrameData& frame_data,
                    const ModelData& model_data,
                    const CameraParameters& cam_params,
                    const int pyramid_height,
                    const float distance_threshold, 
                    const float angle_threshold,
                    const std::vector<int>& iterations);

void estimate_step_multi_threads(const Eigen::Matrix3f& rotation_current, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current,
                const cv::Mat& vertex_map_current, 
                const cv::Mat& normal_map_current,
                const Eigen::Matrix3f& rotation_previous_inv, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous,
                const CameraParameters& cam_params,
                const cv::Mat& vertex_map_previous, 
                const cv::Mat& normal_map_previous,
                float distance_threshold, 
                float angle_threshold,
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, 
                Eigen::Matrix<double, 6, 1>& b);

void estimate_step_pixel_slice(int cols, int rows, int rowStart, int rowEnd,
                const Eigen::Matrix3f& rotation_current, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current,
                const cv::Mat& vertex_map_current, 
                const cv::Mat& normal_map_current,
                const Eigen::Matrix3f& rotation_previous_inv, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous,
                const CameraParameters& cam_params,
                const cv::Mat& vertex_map_previous, 
                const cv::Mat& normal_map_previous,
                float distance_threshold, 
                float angle_threshold,
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, 
                Eigen::Matrix<double, 6, 1>& b);

void estimate_step(const Eigen::Matrix3f& rotation_current, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current,
                const cv::Mat& vertex_map_current, 
                const cv::Mat& normal_map_current,
                const Eigen::Matrix3f& rotation_previous_inv, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous,
                const CameraParameters& cam_params,
                const cv::Mat& vertex_map_previous, 
                const cv::Mat& normal_map_previous,
                float distance_threshold, 
                float angle_threshold,
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, 
                Eigen::Matrix<double, 6, 1>& b);