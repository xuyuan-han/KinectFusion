#pragma once
#include "pose_estimation.hpp"
#include "surface_measurement.hpp"
#include "surface_prediction.hpp"
#include "surface_reconstruction.hpp"
#include "VirtualSensor.hpp"
#include "data_types.hpp"
#include <opencv2/viz/viz3d.hpp>

class Pipeline {
public:
    Pipeline(const CameraParameters _camera_parameters, const GlobalConfiguration _configuration);

    ~Pipeline() = default;

    bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map);
    std::vector<Eigen::Matrix4f> get_poses() const;
    cv::Mat get_last_model_color_frame() const;
    cv::Mat get_last_model_vertex_frame() const;
    cv::Mat get_last_model_normal_frame() const;
    
private:
    const CameraParameters camera_parameters;
    const GlobalConfiguration configuration;
    Volume volume;
    VolumeData volumedata;
    ModelData model_data;
    Eigen::Matrix4f current_pose;
    std::vector<Eigen::Matrix4f> poses;
    cv::Mat last_model_color_frame;
    cv::Mat last_model_vertex_frame;
    cv::Mat last_model_normal_frame;
    size_t frame_id { 0 };
};