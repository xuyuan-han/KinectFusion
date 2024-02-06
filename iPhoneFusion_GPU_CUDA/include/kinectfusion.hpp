#pragma once
#include "data_types_CPU.hpp"
#include "VirtualSensor.hpp"
#include "pose_estimation.hpp"
#include "surface_measurement.hpp"
#include "surface_prediction.hpp"
#include "surface_reconstruction.hpp"
// #include "data_types_GPU.hpp"
// #include <opencv2/viz/viz3d.hpp>

// #ifdef HAS_RECORD3D
// #include "iPhoneFusion.hpp"
// #endif

class Pipeline {
public:
    Pipeline(CameraParameters _camera_parameters, const GlobalConfiguration _configuration);

    ~Pipeline() = default;

    bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, CameraParameters _camera_parameters);
    std::vector<Eigen::Matrix4f> get_poses() const;
    cv::Mat get_last_model_color_frame() const;
    cv::Mat get_last_model_normal_frame_in_camera_coordinates() const;
    void save_tsdf_color_volume_point_cloud() const;
    
private:
    CameraParameters camera_parameters;
    const GlobalConfiguration configuration;
    GPU::VolumeData volume_data_GPU;
    GPU::ModelData model_data_GPU;
    Eigen::Matrix4f current_pose;
    std::vector<Eigen::Matrix4f> poses;
    cv::Mat last_model_color_frame;
    cv::Mat last_model_normal_frame;
    size_t frame_id { 0 };
};

struct Point {
    float x, y, z;
    uint8_t r, g, b; // Color
};

void createAndSaveTSDFPointCloudVolumeData_multi_threads(const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, int3 volume_size_int3, float voxel_scale, float truncation_distance, bool showFaces);

void saveTSDFPointCloudProcessVolumeSlice(const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, float truncation_distance, bool showFaces);

void createAndSaveColorPointCloudVolumeData_multi_threads(const cv::Mat& colorMatrix, const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, int3 volume_size_int3, float voxel_scale, bool showFaces);

void saveColorPointCloudProcessVolumeSlice(const cv::Mat& colorMatrix, const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, bool showFaces);

int save_camera_pose_point_cloud(Eigen::Matrix4f current_pose, int numVertices, std::string outputFilename);

cv::Mat rotate_map_multi_threads(const cv::Mat& mat, const Eigen::Matrix3f& rotation);

void rotate_map_MatSlice(cv::Mat& mat, const Eigen::Matrix3f& rotation, int startRow, int endRow);