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
    // current_pose(0, 3) = _configuration.volume_size[0] / 2 * _configuration.voxel_scale;
    // current_pose(1, 3) = _configuration.volume_size[1] / 2 * _configuration.voxel_scale;
    // current_pose(2, 3) = _configuration.volume_size[2] / 2 * _configuration.voxel_scale - _configuration.init_depth;
    current_pose(0, 3) = 0;
    current_pose(1, 3) = 0;
    current_pose(2, 3) = 0-_configuration.init_depth;
}

bool Pipeline::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map)
{
    std::cout << ">> 1 Surface measurement begin" << std::endl;

    FrameData frame_data = surface_measurement(
        depth_map,
        camera_parameters,
        configuration.num_levels,
        configuration.depth_cutoff_distance,
        configuration.bfilter_kernel_size,
        configuration.bfilter_color_sigma,
        configuration.bfilter_spatial_sigma);
    frame_data.color_pyramid[0] = color_map;
    // std::cout << frame_data.depth_pyramid[0] << std::endl;


    // std::cout << "Pose before ICP: \n" << current_pose << std::endl;
    std::cout << ">>> 1 Surface measurement done" << std::endl;

    std::cout << ">> 2 Pose estimation begin" << std::endl;

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

    std::cout << ">>> 2 Pose estimation done" << std::endl;

    std::cout << ">> 3 Surface reconstruction begin" << std::endl;

    Surface_Reconstruction::integrate(
        frame_data.depth_pyramid[0],
        frame_data.color_pyramid[0],
        &volume,
        camera_parameters,
        configuration.truncation_distance,
        current_pose);

    std::cout << ">>> 3 Surface reconstruction done" << std::endl;

    std::cout << ">> 3.5 Point cloud generation begin" << std::endl;

    // volumedata.tsdf_volume = volume.getVolume();
    // volumedata.color_volume = volume.getColorVolume();

    // createAndSavePointCloud(volumedata.tsdf_volume, "pointcloud.ply", configuration.volume_size);

    volumedata.tsdf_volume = volume.getVolumeData();
    volumedata.color_volume = volume.getColorVolumeData();

    createAndSavePointCloudVolumeData(volumedata.tsdf_volume, current_pose, "pointcloud.ply", configuration.volume_size, true);

    std::cout << ">>> 3.5 Point cloud generation done" << std::endl;

    std::cout << ">> 4 Surface prediction begin" << std::endl;

    for (int level = 0; level < configuration.num_levels; ++level){
        std::cout << ">> 4 (level)" << level << " Surface prediction begin" << std::endl;
        surface_prediction(
            volumedata,
            model_data.vertex_pyramid[level],
            model_data.normal_pyramid[level],
            model_data.color_pyramid[level],
            camera_parameters.level(level),
            configuration.truncation_distance,
            current_pose);
        std::cout << ">> 4 (level)" << level << " Surface prediction done" << std::endl;
    }

    std::cout << ">>> 4 Surface prediction done" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    last_model_color_frame = model_data.color_pyramid[0];
    last_model_vertex_frame = model_data.vertex_pyramid[0];
    last_model_normal_frame = model_data.normal_pyramid[0];
    ++frame_id;
    return true;
}

std::vector<Eigen::Matrix4f> Pipeline::get_poses() const
{
    for (auto pose : poses)
        pose.block(0, 0, 3, 3) = pose.block(0, 0, 3, 3).inverse();
    return poses;
}

cv::Mat Pipeline::get_last_model_color_frame() const
{
    return last_model_color_frame;
}

cv::Mat Pipeline::get_last_model_vertex_frame() const
{
    return last_model_vertex_frame;
}

cv::Mat Pipeline::get_last_model_normal_frame() const
{
    return last_model_normal_frame;
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

    const float tsdf_min = -25.0f; // Minimum TSDF value
    const float tsdf_max = 25.0f;  // Maximum TSDF value

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = 0; k < dz; ++k) {
                // Retrieve the TSDF value
                short tsdfValue = tsdfMatrix.at<short>(j * dz + k, i, 0);

                if (abs(tsdfValue) > 25 || tsdfValue == 0) {
                    // Skip invalid TSDF values
                    continue;
                }

                // Normalize the TSDF value to a 0-1 range
                float normalized_tsdf = (tsdfValue - tsdf_min) / (tsdf_max - tsdf_min);

                Point point;
                point.x = i;
                point.y = j;
                point.z = k;

                // Interpolate between magenta (low TSDF) and green (high TSDF) based on normalized_tsdf
                point.r = static_cast<unsigned char>((1.0f - normalized_tsdf) * 255); // Magenta component decreases with TSDF
                point.g = static_cast<unsigned char>(normalized_tsdf * 255); // Green component increases with TSDF
                point.b = static_cast<unsigned char>((1.0f - normalized_tsdf) * 255); // Magenta component decreases with TSDF

                // Write the point
                tempFile << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << "\n";

                // Increment the vertex count
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

void createAndSavePointCloudVolumeData(const cv::Mat& tsdfMatrix, Eigen::Matrix4f current_pose, const std::string& outputFilename, Eigen::Vector3i volume_size, bool showFaces) {
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

    const float tsdf_min = -25.0f; // Minimum TSDF value
    const float tsdf_max = 25.0f;  // Maximum TSDF value

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = 0; k < dz; ++k) {
                if ( (i==0 || j==0 || k==0 || i==dx-1 || j==dy-1 || k==dz-1) && showFaces && (i%4==3 && j%4==3 && k%4==3)){
                    Point point;
                    point.x = i;
                    point.y = j;
                    point.z = k;

                    // show the faces of the volume
                    point.r = static_cast<unsigned char>(255);
                    point.g = static_cast<unsigned char>(255);
                    point.b = static_cast<unsigned char>(255);

                    // Write the point
                    tempFile << point.x << " " << point.y << " " << point.z << " "
                            << static_cast<int>(point.r) << " "
                            << static_cast<int>(point.g) << " "
                            << static_cast<int>(point.b) << "\n";

                    // Increment the vertex count
                    ++numVertices;
                }
                else
                {
                    // Retrieve the TSDF value
                    short tsdfValue = tsdfMatrix.at<cv::Vec<short, 2>>(j * dz + k, i)[0];
                    short weight = tsdfMatrix.at<cv::Vec<short, 2>>(j * dz + k, i)[1];

                    // if (tsdfValue != 0){
                    //     std::cout << "(tsdfValue, weight): (" << tsdfValue << ", " << weight << ")" << std::endl;
                    // }

                    if (abs(tsdfValue) > 25 || tsdfValue == 0) {
                        // Skip invalid TSDF values
                        continue;
                    }

                    // Normalize the TSDF value to a 0-1 range
                    float normalized_tsdf = (tsdfValue - tsdf_min) / (tsdf_max - tsdf_min);

                    Point point;
                    point.x = i;
                    point.y = j;
                    point.z = k;

                    // Interpolate between magenta (low TSDF) and green (high TSDF) based on normalized_tsdf
                    point.r = static_cast<unsigned char>((1.0f - normalized_tsdf) * 255); // Magenta component decreases with TSDF
                    point.g = static_cast<unsigned char>(normalized_tsdf * 255); // Green component increases with TSDF
                    point.b = static_cast<unsigned char>((1.0f - normalized_tsdf) * 255); // Magenta component decreases with TSDF
 
                    // Write the point
                    tempFile << point.x << " " << point.y << " " << point.z << " "
                            << static_cast<int>(point.r) << " "
                            << static_cast<int>(point.g) << " "
                            << static_cast<int>(point.b) << "\n";

                    // Increment the vertex count
                    ++numVertices;
                }
            }
        }
    }
    
    // Show the camera pose in ply
    // Camera pyramid size
    const float pyramidBaseSize = 100.f; // size of base
    const float pyramidHeight = 200.f;   // height of pyramid

    // The base vertex of the pyramid, relative to the camera center
    Eigen::Matrix<float, 4, 3> pyramidBase;
    pyramidBase <<
        -pyramidBaseSize, -pyramidBaseSize, pyramidHeight,
        pyramidBaseSize, -pyramidBaseSize, pyramidHeight,
        pyramidBaseSize, pyramidBaseSize, pyramidHeight,
        -pyramidBaseSize, pyramidBaseSize, pyramidHeight;

    // apex of pyramid
    Eigen::Vector3f pyramidApex(0, 0, 0);

    // Transform base and vertices to world coordinate system
    for (int i = 0; i < 4; ++i) {
        pyramidBase.row(i) = (current_pose * Eigen::Vector4f(pyramidBase.row(i).x(), pyramidBase.row(i).y(), pyramidBase.row(i).z(), 1)).head<3>();
    }
    pyramidApex = (current_pose * Eigen::Vector4f(pyramidApex.x(), pyramidApex.y(), pyramidApex.z(), 1)).head<3>();

    // Write the pyramid vertices to the file
    for (int i = 0; i < 4; ++i) {
        tempFile << pyramidBase(i, 0) << " " << pyramidBase(i, 1) << " " << pyramidBase(i, 2) << " 0 0 255\n"; // blue base
        ++numVertices;
    }
    tempFile << pyramidApex.x() << " " << pyramidApex.y() << " " << pyramidApex.z() << " 255 0 0\n"; // red apex
    ++numVertices;

    const int lineResolution = 50; // Number of points along each line

    // Generate and write points along the edges of the pyramid base
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3f baseVertexStart = pyramidBase.row(i);
        Eigen::Vector3f baseVertexEnd = pyramidBase.row((i + 1) % 4); // 循环连接到下一个顶点

        for (int j = 0; j <= lineResolution; ++j) {
            float t = static_cast<float>(j) / static_cast<float>(lineResolution);
            Eigen::Vector3f pointOnBaseLine = baseVertexStart + t * (baseVertexEnd - baseVertexStart);

            // Write the point on the base edge line
            tempFile << pointOnBaseLine.x() << " " << pointOnBaseLine.y() << " " << pointOnBaseLine.z() << " 255 255 0\n"; // yellow line
            ++numVertices;
        }
    }

    // Generate and write points along the lines from the pyramid base to the apex
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3f baseVertex = pyramidBase.row(i);
        for (int j = 0; j <= lineResolution; ++j) {
            float t = static_cast<float>(j) / static_cast<float>(lineResolution);
            Eigen::Vector3f pointOnLine = baseVertex + t * (pyramidApex - baseVertex);

            // Write the point on line to the apex
            tempFile << pointOnLine.x() << " " << pointOnLine.y() << " " << pointOnLine.z() << " 255 255 0\n"; // yellow line
            ++numVertices;
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