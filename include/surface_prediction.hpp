#pragma once
#include "data_types.hpp"

float interpolate_trilinearly(
    Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& point,
    cv::Mat& volume,  // TSDF Volume
    const Eigen::Vector3f& volume_size,
    const float voxel_scale);

float get_min_time(
    const Eigen::Vector3f& volume_max,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& origin,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& direction);

float get_max_time(
    const Eigen::Vector3f& volume_max,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& origin,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& direction
);

void raycast_tsdf_kernel(
    cv::Mat& tsdf_volume,                       // global TSDF Volume
    cv::Mat& color_volume,                      // global color Volume
    cv::Mat& model_vertex,                       
    cv::Mat& model_normal,                        
    cv::Mat& model_color,
    const Eigen::Vector3i volume_size,
    const float voxel_scale,
    const CameraParameters& cam_parameters,
    const float truncation_distance,
    const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> translation
);

void surface_prediction(
    VolumeData& volume,                   // Global Volume
    cv::Mat& model_vertex,                       
    cv::Mat& model_normal,                       
    cv::Mat& model_color,                        
    const CameraParameters& cam_parameters,     
    const float truncation_distance,            
    const Eigen::Matrix4f& pose);

void raycast_tsdf_kernel_volume_slice(    
    cv::Mat& tsdf_volume,                       // global TSDF Volume
    cv::Mat& color_volume,                      // global color Volume
    cv::Mat& model_vertex,                       
    cv::Mat& model_normal,                        
    cv::Mat& model_color,
    const Eigen::Vector3i volume_size,
    const float voxel_scale,
    const CameraParameters& cam_parameters,
    const float truncation_distance,
    const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> translation,
    int rowStart, int rowEnd);