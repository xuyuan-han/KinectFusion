#pragma once
#include "data_types_CPU.hpp"
#include "VirtualSensor.hpp"
#include "pose_estimation.hpp"
#include "surface_measurement.hpp"
#include "surface_prediction.hpp"
#include "surface_reconstruction.hpp"
// #include "data_types_GPU.hpp"
// #include <opencv2/viz/viz3d.hpp>

class Pipeline {
public:
    Pipeline(const CameraParameters _camera_parameters, const GlobalConfiguration _configuration, const std::string _outputPath);

    ~Pipeline() = default;

    bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map);
    std::vector<Eigen::Matrix4f> get_poses() const;
    cv::Mat get_last_model_color_frame() const;
    cv::Mat get_last_model_normal_frame() const;
    cv::Mat get_last_model_vertex_frame() const;
    cv::Mat get_last_model_normal_frame_in_camera_coordinates() const;
    void save_tsdf_color_volume_point_cloud() const;

    #ifdef SHOW_STATIC_CAMERA_MODEL
    cv::Mat get_static_last_model_color_frame() const;
    cv::Mat get_static_last_model_normal_frame() const;
    cv::Mat get_static_last_model_vertex_frame() const;
    #endif
    
private:
    const CameraParameters camera_parameters;
    const GlobalConfiguration configuration;
    std::string outputPath;
    GPU::VolumeData volume_data_GPU;
    GPU::ModelData model_data_GPU;
    Eigen::Matrix4f current_pose;
    std::vector<Eigen::Matrix4f> poses;
    cv::Mat last_model_color_frame;
    cv::Mat last_model_normal_frame;
    cv::Mat last_model_vertex_frame;

    #ifdef SHOW_STATIC_CAMERA_MODEL
    GPU::ModelData static_model_data_GPU;
    cv::Mat static_last_model_color_frame;
    cv::Mat static_last_model_normal_frame;
    cv::Mat static_last_model_vertex_frame;
    #endif

    size_t frame_id { 0 };
};

struct Point {
    float x, y, z;
    uint8_t r, g, b; // Color
};

void createAndSaveTSDFPointCloudVolumeData_multi_threads(const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, int3 volume_size_int3, float voxel_scale, float truncation_distance, bool showVolumeCorners);

void saveTSDFPointCloudProcessVolumeSlice(const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, float truncation_distance, bool showVolumeCorners);

void createAndSaveColorPointCloudVolumeData_multi_threads(const cv::Mat& colorMatrix, const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, int3 volume_size_int3, float voxel_scale, bool showVolumeCorners);

void saveColorPointCloudProcessVolumeSlice(const cv::Mat& colorMatrix, const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, bool showVolumeCorners);

int save_camera_pose_point_cloud(Eigen::Matrix4f current_pose, int numVertices, std::string outputFilename);

cv::Mat rotate_map_multi_threads(const cv::Mat& mat, const Eigen::Matrix3f& rotation);

void rotate_map_MatSlice(cv::Mat& mat, const Eigen::Matrix3f& rotation, int startRow, int endRow);

cv::Mat normalMapping(const cv::Mat& normal, const cv::Vec3f& lightPosition, const cv::Mat& vectex);

void processNormalMapping(const cv::Mat& normal, const cv::Vec3f& lightPosition, const cv::Mat& vertex, cv::Mat& results, int startRow, int endRow);

std::string getCurrentTimestamp();