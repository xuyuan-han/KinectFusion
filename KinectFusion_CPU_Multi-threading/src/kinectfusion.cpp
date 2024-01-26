#include "kinectfusion.hpp"

Pipeline::Pipeline(const CameraParameters _camera_parameters,
                    const GlobalConfiguration _configuration) :
        camera_parameters(_camera_parameters),
        configuration(_configuration),
        volumedata(_configuration.volume_size, _configuration.voxel_scale),
        model_data(_configuration.num_levels, _camera_parameters),
        current_pose{},
        poses{},
        frame_id{0}
{
    current_pose.setIdentity();
    current_pose(0, 3) = _configuration.volume_size[0] / 2 * _configuration.voxel_scale;
    current_pose(1, 3) = 0;
    current_pose(2, 3) = _configuration.volume_size[2] / 2 * _configuration.voxel_scale - _configuration.init_depth;

    float beta = -40.0f; // rotate around x axis by -45 degrees
    beta = beta / 180.0f * M_PI;
    Eigen::Matrix3f rotation_matrix;
    rotation_matrix << 1, 0, 0,
                       0, cosf(beta), -sinf(beta),
                       0, sinf(beta), cosf(beta);
    current_pose.block(0, 0, 3, 3) = rotation_matrix;
}

bool Pipeline::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map)
{
    auto start = std::chrono::high_resolution_clock::now();
    FrameData frame_data = surface_measurement(
        depth_map,
        camera_parameters,
        configuration.num_levels,
        configuration.depth_cutoff_distance,
        configuration.bfilter_kernel_size,
        configuration.bfilter_color_sigma,
        configuration.bfilter_spatial_sigma);
    frame_data.color_pyramid[0] = color_map;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "-- Surface measurement:\t" << elapsed.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
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
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "-- Pose estimation:\t" << elapsed.count() << " ms" << std::endl;

    if (!icp_success)
        return false;
    poses.push_back(current_pose);

    start = std::chrono::high_resolution_clock::now();
#ifdef USE_CPU_MULTI_THREADING
    Surface_Reconstruction::integrate_multi_threads(
        frame_data.depth_pyramid[0],
        frame_data.color_pyramid[0],
        &volumedata,
        camera_parameters,
        configuration.truncation_distance,
        current_pose);
#else
    Surface_Reconstruction::integrate(
        frame_data.depth_pyramid[0],
        frame_data.color_pyramid[0],
        &volumedata,
        camera_parameters,
        configuration.truncation_distance,
        current_pose);
#endif
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "-- Surface reconstruct: " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int level = 0; level < configuration.num_levels; ++level){
        surface_prediction(
            volumedata,
            model_data.vertex_pyramid[level],
            model_data.normal_pyramid[level],
            model_data.color_pyramid[level],
            camera_parameters.level(level),
            configuration.truncation_distance,
            current_pose);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "-- Surface prediction:\t" << elapsed.count() << " ms" << std::endl;

    last_model_color_frame = model_data.color_pyramid[0];
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

cv::Mat Pipeline::get_last_model_normal_frame() const
{
    return last_model_normal_frame;
}

cv::Mat Pipeline::get_last_model_normal_frame_in_camera_coordinates() const
{
    return rotate_map_multi_threads(last_model_normal_frame, current_pose.block(0, 0, 3, 3).inverse());
}

void Pipeline::save_tsdf_color_volume_point_cloud() const
{
    createAndSavePointCloudVolumeData_multi_threads(volumedata.tsdf_volume, poses, "TSDF_VolumeData_PointCloud.ply", configuration.volume_size, configuration.voxel_scale, configuration.truncation_distance, true);
    createAndSaveColorPointCloudVolumeData_multi_threads(volumedata.color_volume, poses, "Color_VolumeData_PointCloud.ply", configuration.volume_size, configuration.voxel_scale, true);
}

// multi threads version
void createAndSavePointCloudVolumeData_multi_threads(const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, Eigen::Vector3i volume_size, float voxel_scale, float truncation_distance, bool showFaces) {
    // Keep track of the number of vertices
    int numVertices = 0;
    int dx = volume_size[0];
    int dy = volume_size[1];
    int dz = volume_size[2];

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    int zStep = dz / numThreads;
    std::vector<int> numVerticesVec(numThreads, 0);
    std::vector<std::string> tempFilenames(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        tempFilenames[i] = "temp_" + std::to_string(i) + ".ply";
        int zStart = i * zStep;
        int zEnd = (i + 1) * zStep;
        if (i == numThreads - 1) zEnd = dz;
        threads[i] = std::thread(savePointCloudProcessVolumeSlice, std::ref(tsdfMatrix), tempFilenames[i], dx, dy, dz, zStart, zEnd, std::ref(numVerticesVec[i]), voxel_scale, truncation_distance, showFaces);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int nv : numVerticesVec) {
        numVertices += nv;
    }
    
    std::vector<std::string> camera_pose_tempFilenames(poses.size());
    for (int i = 0; i < poses.size(); ++i) {
        // save the camera pose in the .ply file
        camera_pose_tempFilenames[i] = "camera_pose_tempFile_" + std::to_string(i) + ".ply";
        numVertices = save_camera_pose_point_cloud(poses[i], numVertices, camera_pose_tempFilenames[i]);
    }

    std::ofstream plyFile(outputFilename);
    if (!plyFile.is_open()) {
        std::cerr << "Unable to open file: " << outputFilename << std::endl;
        return;
    }

    plyFile << "ply\nformat ascii 1.0\n";
    plyFile << "element vertex " << numVertices << "\n";
    plyFile << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    plyFile << "end_header\n";

    for (const auto& tempFilename : tempFilenames) {
        std::ifstream tempFile(tempFilename);
        if (tempFile.good()) {
            plyFile << tempFile.rdbuf();
            tempFile.close();
            std::remove(tempFilename.c_str());
        } else {
            std::cerr << "Error reading temporary file: " << tempFilename << std::endl;
        }
    }

    for (const auto& camera_pose_tempFilename : camera_pose_tempFilenames) {
        std::ifstream tempFilePyramid(camera_pose_tempFilename);
        if (tempFilePyramid.good()) {
            plyFile << tempFilePyramid.rdbuf();
            tempFilePyramid.close();
            std::remove(camera_pose_tempFilename.c_str());
        } else {
            std::cerr << "Error reading temporary file: " << camera_pose_tempFilename << std::endl;
        }
    }

    plyFile.close();
}

void savePointCloudProcessVolumeSlice(const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, float truncation_distance, bool showFaces) {
    std::ofstream tempFile(tempFilename);
    if (!tempFile.is_open()) {
        std::cerr << "Unable to open temporary file: " << tempFilename << std::endl;
        return;
    }

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = zStart; k < zEnd; ++k) {
                if ( (i==0 || j==0 || k==0 || i==dx-1 || j==dy-1 || k==dz-1) && showFaces && (i%4==3 && j%4==3 && k%4==3)){
                    Point point;
                    point.x = i * voxel_scale;
                    point.y = j * voxel_scale;
                    point.z = k * voxel_scale;

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
                    float tsdfValue = tsdfMatrix.at<cv::Vec<short, 2>>(k * dy + j, i)[0] * DIVSHORTMAX;
                    short weight = tsdfMatrix.at<cv::Vec<short, 2>>(k * dy + j, i)[1];

                    if (abs(tsdfValue) < 1.0f - 1e-4 && tsdfValue != 0){
                        // Normalize the TSDF value to a 0-1 range
                        float normalized_tsdf = (tsdfValue - (-1.0f)) / (1.0f - (-1.0f));

                        Point point;
                        point.x = i * voxel_scale;
                        point.y = j * voxel_scale;
                        point.z = k * voxel_scale;

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
    }
    tempFile.close();
}

// multi threads version
void createAndSaveColorPointCloudVolumeData_multi_threads(const cv::Mat& colorMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, Eigen::Vector3i volume_size, float voxel_scale, bool showFaces) {
    // Keep track of the number of vertices
    int numVertices = 0;
    int dx = volume_size[0];
    int dy = volume_size[1];
    int dz = volume_size[2];

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    int zStep = dz / numThreads;
    std::vector<int> numVerticesVec(numThreads, 0);
    std::vector<std::string> tempFilenames(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        tempFilenames[i] = "temp_" + std::to_string(i) + ".ply";
        int zStart = i * zStep;
        int zEnd = (i + 1) * zStep;
        if (i == numThreads - 1) zEnd = dz;
        threads[i] = std::thread(saveColorPointCloudProcessVolumeSlice, std::ref(colorMatrix), tempFilenames[i], dx, dy, dz, zStart, zEnd, std::ref(numVerticesVec[i]), voxel_scale, showFaces);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int nv : numVerticesVec) {
        numVertices += nv;
    }

    std::vector<std::string> camera_pose_tempFilenames(poses.size());
    for (int i = 0; i < poses.size(); ++i) {
        // save the camera pose in the .ply file
        camera_pose_tempFilenames[i] = "camera_pose_tempFile_" + std::to_string(i) + ".ply";
        numVertices = save_camera_pose_point_cloud(poses[i], numVertices, camera_pose_tempFilenames[i]);
    }

    std::ofstream plyFile(outputFilename);
    if (!plyFile.is_open()) {
        std::cerr << "Unable to open file: " << outputFilename << std::endl;
        return;
    }

    plyFile << "ply\nformat ascii 1.0\n";
    plyFile << "element vertex " << numVertices << "\n";
    plyFile << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    plyFile << "end_header\n";

    for (const auto& tempFilename : tempFilenames) {
        std::ifstream tempFile(tempFilename);
        if (tempFile.good()) {
            plyFile << tempFile.rdbuf();
            tempFile.close();
            std::remove(tempFilename.c_str());
        } else {
            std::cerr << "Error reading temporary file: " << tempFilename << std::endl;
        }
    }

    for (const auto& camera_pose_tempFilename : camera_pose_tempFilenames) {
        std::ifstream tempFilePyramid(camera_pose_tempFilename);
        if (tempFilePyramid.good()) {
            plyFile << tempFilePyramid.rdbuf();
            tempFilePyramid.close();
            std::remove(camera_pose_tempFilename.c_str());
        } else {
            std::cerr << "Error reading temporary file: " << camera_pose_tempFilename << std::endl;
        }
    }

    plyFile.close();
}

void saveColorPointCloudProcessVolumeSlice(const cv::Mat& colorMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, bool showFaces) {
    std::ofstream tempFile(tempFilename);
    if (!tempFile.is_open()) {
        std::cerr << "Unable to open temporary file: " << tempFilename << std::endl;
        return;
    }

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = zStart; k < zEnd; ++k) {
                if ( (i==0 || j==0 || k==0 || i==dx-1 || j==dy-1 || k==dz-1) && showFaces && (i%4==3 && j%4==3 && k%4==3)){
                    Point point;
                    point.x = i * voxel_scale;
                    point.y = j * voxel_scale;
                    point.z = k * voxel_scale;

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
                    // Retrieve the color value
                    cv::Vec3b colorValue = colorMatrix.at<cv::Vec3b>(k * dy + j, i);

                    if (colorValue == cv::Vec3b{0, 0, 0}) {
                        // Skip invalid color values
                        continue;
                    }

                    Point point;
                    point.x = i * voxel_scale;
                    point.y = j * voxel_scale;
                    point.z = k * voxel_scale;

                    point.r = colorValue[2];
                    point.g = colorValue[1];
                    point.b = colorValue[0];
 
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
    tempFile.close();
}

int save_camera_pose_point_cloud(Eigen::Matrix4f current_pose, int numVertices, std::string outputFilename){
    // Show the camera pose in the .ply file
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

    std::ofstream tempFilePyramid(outputFilename);

    // Write the pyramid vertices to the file
    for (int i = 0; i < 4; ++i) {
        tempFilePyramid << pyramidBase(i, 0) << " " << pyramidBase(i, 1) << " " << pyramidBase(i, 2) << " 0 0 255\n"; // blue base
        ++numVertices;
    }
    tempFilePyramid << pyramidApex.x() << " " << pyramidApex.y() << " " << pyramidApex.z() << " 255 0 0\n"; // red apex
    ++numVertices;

    const int lineResolution = 50; // Number of points along each line

    // Generate and write points along the edges of the pyramid base
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3f baseVertexStart = pyramidBase.row(i);
        Eigen::Vector3f baseVertexEnd = pyramidBase.row((i + 1) % 4);

        for (int j = 0; j <= lineResolution; ++j) {
            float t = static_cast<float>(j) / static_cast<float>(lineResolution);
            Eigen::Vector3f pointOnBaseLine = baseVertexStart + t * (baseVertexEnd - baseVertexStart);

            // Write the point on the base edge line
            tempFilePyramid << pointOnBaseLine.x() << " " << pointOnBaseLine.y() << " " << pointOnBaseLine.z() << " 255 255 0\n"; // yellow line
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
            tempFilePyramid << pointOnLine.x() << " " << pointOnLine.y() << " " << pointOnLine.z() << " 255 255 0\n"; // yellow line
            ++numVertices;
        }
    }
    tempFilePyramid.close();
    return numVertices;
}

// can be used to rotate the model normal map to the camera coordinate system
cv::Mat rotate_map_multi_threads(const cv::Mat& mat, const Eigen::Matrix3f& rotation) {
    cv::Mat matCopy = mat.clone();
    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    int numRowsPerThread = mat.rows / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * numRowsPerThread;
        int endRow = (i + 1) * numRowsPerThread;

        if (i == numThreads - 1) {
            endRow = mat.rows;
        }

        threads[i] = std::thread(rotate_map_MatSlice, std::ref(matCopy), std::ref(rotation), startRow, endRow);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return matCopy;
}

void rotate_map_MatSlice(cv::Mat& mat, const Eigen::Matrix3f& rotation, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            cv::Vec3f& pixel = mat.at<cv::Vec3f>(i, j);
            Eigen::Vector3f vec(pixel[0], pixel[1], pixel[2]);
            Eigen::Vector3f rotatedVec = rotation * vec;
            pixel = cv::Vec3f(rotatedVec[0], rotatedVec[1], rotatedVec[2]);
        }
    }
}