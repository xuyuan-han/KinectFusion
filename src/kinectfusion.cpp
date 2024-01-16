#include "kinectfusion.hpp"

Pipeline::Pipeline(const CameraParameters _camera_parameters,
                    const GlobalConfiguration _configuration) :
        camera_parameters(_camera_parameters),
        configuration(_configuration),
        volume(_configuration.volume_size, _configuration.voxel_scale),
        volumedata(_configuration.volume_size, _configuration.voxel_scale),
        model_data(_configuration.num_levels, _camera_parameters),
        current_pose{},
        poses{},
        frame_id{0}
{
    current_pose.setIdentity();
    current_pose(0, 3) = _configuration.volume_size[0] / 2 * _configuration.voxel_scale;
    current_pose(1, 3) = _configuration.volume_size[1] / 2 * _configuration.voxel_scale;
    current_pose(2, 3) = _configuration.volume_size[2] / 2 * _configuration.voxel_scale - _configuration.init_depth;
}
struct Point {
    float x, y, z;
    uint8_t r, g, b; // Color
};
void createAndSavePointCloud(const cv::Mat& tsdfMatrix, const std::string& outputFilename, Eigen::Vector3i volume_size) {
    std::ofstream plyFile(outputFilename);

    if (!plyFile.is_open()) {
        std::cerr << "Unable to open file: " << outputFilename << std::endl;
        return;
    }

    // Write to something temporary and then copy to the final file
    // This is done to update the number of vertices in the header
    std::ofstream tempFile("temp.ply");
    // Keep track of the number of vertices
    int numVertices = 0;
    int dx = volume_size[0];
    int dy = volume_size[1];
    int dz = volume_size[2];

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = 0; k < dz; ++k) {
                // Get TSDF value
                short tsdfValue = tsdfMatrix.at<short>(j * dz + k, i, 0);

                if (abs(tsdfValue) > 25 || tsdfValue == 0) {
                    // Skip invalid TSDF values
                    continue;
                }

                Point point;

                // Set point coordinates
                point.x = i;
                point.y = j;
                point.z = k;


                // Set point color based on TSDF value
                // Green for negative TSDF
                point.r = 0;
                point.g = 255;
                point.b = 0;

                // Write point
                tempFile << point.x << " " << point.y << " " << point.z << " " << static_cast<int>(point.r) << " " << static_cast<int>(point.g) << " " << static_cast<int>(point.b) << "\n";

                // Increment the number of vertices
                ++numVertices;
            }
        }
    }

    // Copy the temporary file to the final file and update the header
    tempFile.close();
    tempFile.open("temp.ply", std::ios::in);
    plyFile << "ply\nformat ascii 1.0\n";
    plyFile << "element vertex " << numVertices << "\n";
    plyFile << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    plyFile << "end_header\n";
    plyFile << tempFile.rdbuf();

    // Close and remove the temporary file
    tempFile.close();
    std::remove("temp.ply");
}

bool Pipeline::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map)
{
    FrameData frame_data = surface_measurement(
        depth_map,
        camera_parameters,
        configuration.num_levels,
        configuration.depth_cutoff_distance,
        configuration.bfilter_kernel_size,
        configuration.bfilter_color_sigma,
        configuration.bfilter_spatial_sigma);
    frame_data.color_pyramid[0] = color_map;

    std::cout << "Surface measurement done" << std::endl;
    std::cout << "Pose before ICP: " << current_pose << std::endl;
    bool icp_success { true };
    if (frame_id > 0) { // Do not perform ICP for the very first frame
        icp_success = pose_estimation(
            current_pose,
            frame_data,
            model_data,
            camera_parameters,
            configuration.num_levels,
            configuration.distance_threshold,
            configuration.angle_threshold,
            configuration.icp_iterations);
    }
    if (!icp_success)
        return false;
    poses.push_back(current_pose);

    std::cout << "Pose estimation done" << std::endl;

    Surface_Reconstruction::integrate(
        frame_data.depth_pyramid[0],
        frame_data.color_pyramid[0],
        &volume,
        camera_parameters,
        configuration.truncation_distance,
        current_pose);

    
    std::cout << "Surface reconstruction done" << std::endl;


    volumedata.tsdf_volume = volume.getVolume();
    volumedata.color_volume = volume.getColorVolume();

    createAndSavePointCloud(volumedata.tsdf_volume, "pointcloud.ply", configuration.volume_size);

    std::cout << "Point cloud generated" << std::endl;

    for (int level = 0; level < configuration.num_levels; ++level)
        surface_prediction(
            volumedata,
            model_data.vertex_pyramid[level],
            model_data.normal_pyramid[level],
            model_data.color_pyramid[level],
            camera_parameters.level(level),
            configuration.truncation_distance,
            current_pose);
    
    std::cout << "Surface prediction done" << std::endl;

    last_model_frame = model_data.color_pyramid[0];
    ++frame_id;
    return true;
}

std::vector<Eigen::Matrix4f> Pipeline::get_poses() const
{
    for (auto pose : poses)
        pose.block(0, 0, 3, 3) = pose.block(0, 0, 3, 3).inverse();
    return poses;
}

cv::Mat Pipeline::get_last_model_frame() const
{
    return last_model_frame;
}