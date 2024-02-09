#include "kinectfusion.hpp"

Pipeline::Pipeline(const CameraParameters _camera_parameters,
                    const GlobalConfiguration _configuration,
                    const std::string _outputPath) :
        camera_parameters(_camera_parameters),
        configuration(_configuration),
        outputPath(_outputPath),
        volumedata(_configuration.volume_size, _configuration.voxel_scale),
        model_data(_configuration.num_levels, _camera_parameters),
        #ifdef SHOW_STATIC_CAMERA_MODEL
        static_model_data(_configuration.num_levels, _camera_parameters),
        #endif
        current_pose{},
        poses{},
        frame_id{0}
{
    current_pose.setIdentity();
    current_pose(0, 3) = _configuration.volume_size[0] / 2 * _configuration.voxel_scale - _configuration.init_depth_x;
    current_pose(1, 3) = _configuration.volume_size[1] / 2 * _configuration.voxel_scale - _configuration.init_depth_y;
    current_pose(2, 3) = _configuration.volume_size[2] / 2 * _configuration.voxel_scale - _configuration.init_depth_z;

    float alpha = _configuration.init_alpha;
    float beta = _configuration.init_beta;
    float gamma = _configuration.init_gamma;

    alpha = alpha / 180.0f * M_PI;
    beta = beta / 180.0f * M_PI;
    gamma = gamma / 180.0f * M_PI;

    Eigen::Matrix3f rotation_matrix_z;
    rotation_matrix_z << cosf(alpha), -sinf(alpha), 0,
                        sinf(alpha), cosf(alpha), 0,
                        0, 0, 1;

    Eigen::Matrix3f rotation_matrix_y;
    rotation_matrix_y << cosf(beta), 0, sinf(beta),
                        0, 1, 0,
                        -sinf(beta), 0, cosf(beta);

    Eigen::Matrix3f rotation_matrix_x;
    rotation_matrix_x << 1, 0, 0,
                        0, cosf(gamma), -sinf(gamma),
                        0, sinf(gamma), cosf(gamma);

    Eigen::Matrix3f rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;

    current_pose.block(0, 0, 3, 3) = rotation_matrix;
}

bool Pipeline::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, const cv::Mat_<uchar>& segmentation_map)
{
    #ifdef PRINT_MODULE_COMP_TIME
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    FrameData frame_data = surface_measurement(
        depth_map,
        camera_parameters,
        configuration.num_levels,
        configuration.depth_cutoff_distance,
        configuration.bfilter_kernel_size,
        configuration.bfilter_color_sigma,
        configuration.bfilter_spatial_sigma);
    frame_data.color_pyramid[0] = color_map;

    #ifdef PRINT_MODULE_COMP_TIME
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "-- Surface measurement:\t" << elapsed.count() << " ms" << std::endl;
    #endif

    #ifdef PRINT_MODULE_COMP_TIME
    start = std::chrono::high_resolution_clock::now();
    #endif

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

    #ifdef PRINT_MODULE_COMP_TIME
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "-- Pose estimation:\t" << elapsed.count() << " ms" << std::endl;
    #endif

    if (!icp_success)
        return false;
    poses.push_back(current_pose);

    #ifdef PRINT_MODULE_COMP_TIME
    start = std::chrono::high_resolution_clock::now();
    #endif

#ifdef USE_CPU_MULTI_THREADING
    Surface_Reconstruction::integrate_multi_threads(
        frame_data.depth_pyramid[0],
        frame_data.color_pyramid[0],
        segmentation_map,
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

    #ifdef PRINT_MODULE_COMP_TIME
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "-- Surface reconstruct: " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    #endif

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

    #ifdef PRINT_MODULE_COMP_TIME
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "-- Surface prediction:\t" << elapsed.count() << " ms" << std::endl;
    #endif

    last_model_color_frame = model_data.color_pyramid[0];
    last_model_normal_frame = model_data.normal_pyramid[0];
    last_model_vertex_frame = model_data.vertex_pyramid[0];

    #ifdef SHOW_STATIC_CAMERA_MODEL
    Eigen::Matrix4f static_camera_pose = poses[0];
    static_camera_pose(0, 3) -= 0;
    static_camera_pose(1, 3) -= 1000;
    static_camera_pose(2, 3) -= 1000;
    surface_prediction(
        volumedata,
        static_model_data.vertex_pyramid[0],
        static_model_data.normal_pyramid[0],
        static_model_data.color_pyramid[0],
        camera_parameters,
        configuration.truncation_distance,
        static_camera_pose);
    static_last_model_color_frame = static_model_data.color_pyramid[0];
    static_last_model_normal_frame = static_model_data.normal_pyramid[0];
    static_last_model_vertex_frame = static_model_data.vertex_pyramid[0];
    #endif

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

cv::Mat Pipeline::get_last_model_vertex_frame() const
{
    return last_model_vertex_frame;
}

cv::Mat Pipeline::get_last_model_normal_frame_in_camera_coordinates() const
{
    return rotate_map_multi_threads(last_model_normal_frame, current_pose.block(0, 0, 3, 3).inverse());
}

#ifdef SHOW_STATIC_CAMERA_MODEL
cv::Mat Pipeline::get_static_last_model_color_frame() const
{
    return static_last_model_color_frame;
}
cv::Mat Pipeline::get_static_last_model_normal_frame() const
{
    return static_last_model_normal_frame;
}
cv::Mat Pipeline::get_static_last_model_vertex_frame() const
{
    return static_last_model_vertex_frame;
}
#endif

void Pipeline::save_tsdf_color_volume_point_cloud() const
{
    if (!std::filesystem::exists(outputPath)) {
        try {
            if (!std::filesystem::create_directories(outputPath)) {
                std::cerr << "Failed to create output directory: " << outputPath << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    createAndSaveTSDFPointCloudVolumeData_multi_threads(volumedata.tsdf_volume, poses, outputPath + "PointCloud_TSDF_VolumeData.ply", configuration.volume_size, configuration.voxel_scale, configuration.truncation_distance, true);
    createAndSaveColorPointCloudVolumeData_multi_threads(volumedata.color_volume, volumedata.tsdf_volume, poses, outputPath + "PointCloud_Color_VolumeData.ply", configuration.volume_size, configuration.voxel_scale, true);
    #ifdef USE_CLASSES
    createAndSaveClassPointCloudVolumeData_multi_threads(volumedata.class_volume, volumedata.tsdf_volume, poses, outputPath + "PointCloud_Class_VolumeData.ply", configuration.volume_size, configuration.voxel_scale, true);
    #endif
}

// multi threads version
void createAndSaveTSDFPointCloudVolumeData_multi_threads(const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, Eigen::Vector3i volume_size, float voxel_scale, float truncation_distance, bool showVolumeCorners) {
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
        threads[i] = std::thread(saveTSDFPointCloudProcessVolumeSlice, std::ref(tsdfMatrix), tempFilenames[i], dx, dy, dz, zStart, zEnd, std::ref(numVerticesVec[i]), voxel_scale, truncation_distance, showVolumeCorners);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int nv : numVerticesVec) {
        numVertices += nv;
    }
    
    std::vector<std::string> camera_pose_tempFilenames;
    for (int i = 0; i < poses.size();) {
        // save the camera pose in the .ply file
        std::string camera_pose_tempFilename = "camera_pose_tempFile_" + std::to_string(i) + ".ply";
        camera_pose_tempFilenames.push_back(camera_pose_tempFilename);
        numVertices = save_camera_pose_point_cloud(poses[i], numVertices, camera_pose_tempFilename);
        i += 30; // save every 30 frames
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

    for (size_t i = 0; i < tempFilenames.size(); ++i) {
        if (numVerticesVec[i] > 0){
            std::ifstream tempFile(tempFilenames[i]);
            if (tempFile.good()) {
                plyFile << tempFile.rdbuf();
                tempFile.close();
            } else {
                std::cerr << "Error reading temporary file: " << tempFilenames[i] << std::endl;
            }
        }
        std::remove(tempFilenames[i].c_str());
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

void saveTSDFPointCloudProcessVolumeSlice(const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, float truncation_distance, bool showVolumeCorners) {
    std::ofstream tempFile(tempFilename);
    if (!tempFile.is_open()) {
        std::cerr << "Unable to open temporary file: " << tempFilename << std::endl;
        return;
    }

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = zStart; k < zEnd; ++k) {
                const int segmentLength = std::min({dx, dy, dz}) / 6;
                bool onVolumeCorners = (i <= segmentLength || (dx - 1 - i) <= segmentLength) &&
                                (j <= segmentLength || (dy - 1 - j) <= segmentLength) &&
                                (k <= segmentLength || (dz - 1 - k) <= segmentLength) &&
                                ((i == 0 && j == 0) || (i == 0 && k == 0) || (j == 0 && k == 0) || 
                                (i == dx - 1 && j == dy - 1) || (i == dx - 1 && k == dz - 1) || (j == dy - 1 && k == dz - 1) ||
                                (i == 0 && j == dy - 1) || (i == 0 && k == dz - 1) || (j == 0 && k == dz - 1) || 
                                (i == dx - 1 && j == 0) || (i == dx - 1 && k == 0) || (j == dy - 1 && k == 0));

                if (showVolumeCorners && onVolumeCorners){
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

                    if (abs(tsdfValue) < 0.9999f * DIVSHORTMAX * SHORTMAX && tsdfValue != 0){
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
void createAndSaveColorPointCloudVolumeData_multi_threads(const cv::Mat& colorMatrix, const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, Eigen::Vector3i volume_size, float voxel_scale, bool showVolumeCorners) {
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
        threads[i] = std::thread(saveColorPointCloudProcessVolumeSlice, std::ref(colorMatrix), std::ref(tsdfMatrix), tempFilenames[i], dx, dy, dz, zStart, zEnd, std::ref(numVerticesVec[i]), voxel_scale, showVolumeCorners);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int nv : numVerticesVec) {
        numVertices += nv;
    }

    std::vector<std::string> camera_pose_tempFilenames;
    for (int i = 0; i < poses.size();) {
        // save the camera pose in the .ply file
        std::string camera_pose_tempFilename = "camera_pose_tempFile_" + std::to_string(i) + ".ply";
        camera_pose_tempFilenames.push_back(camera_pose_tempFilename);
        numVertices = save_camera_pose_point_cloud(poses[i], numVertices, camera_pose_tempFilename);
        i += 30; // save every 30 frames
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

    for (size_t i = 0; i < tempFilenames.size(); ++i) {
        if (numVerticesVec[i] > 0){
            std::ifstream tempFile(tempFilenames[i]);
            if (tempFile.good()) {
                plyFile << tempFile.rdbuf();
                tempFile.close();
            } else {
                std::cerr << "Error reading temporary file: " << tempFilenames[i] << std::endl;
            }
        }
        std::remove(tempFilenames[i].c_str());
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

void saveColorPointCloudProcessVolumeSlice(const cv::Mat& colorMatrix, const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, bool showVolumeCorners) {
    std::ofstream tempFile(tempFilename);
    if (!tempFile.is_open()) {
        std::cerr << "Unable to open temporary file: " << tempFilename << std::endl;
        return;
    }

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = zStart; k < zEnd; ++k) {
                const int segmentLength = std::min({dx, dy, dz}) / 6;
                bool onVolumeCorners = (i <= segmentLength || (dx - 1 - i) <= segmentLength) &&
                                (j <= segmentLength || (dy - 1 - j) <= segmentLength) &&
                                (k <= segmentLength || (dz - 1 - k) <= segmentLength) &&
                                ((i == 0 && j == 0) || (i == 0 && k == 0) || (j == 0 && k == 0) || 
                                (i == dx - 1 && j == dy - 1) || (i == dx - 1 && k == dz - 1) || (j == dy - 1 && k == dz - 1) ||
                                (i == 0 && j == dy - 1) || (i == 0 && k == dz - 1) || (j == 0 && k == dz - 1) || 
                                (i == dx - 1 && j == 0) || (i == dx - 1 && k == 0) || (j == dy - 1 && k == 0));

                if (showVolumeCorners && onVolumeCorners){
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
                    float tsdfValue = tsdfMatrix.at<cv::Vec<short, 2>>(k * dy + j, i)[0] * DIVSHORTMAX;

                    if (abs(tsdfValue) < 0.2f * DIVSHORTMAX * SHORTMAX && tsdfValue != 0){
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
        tempFilePyramid << pyramidBase(i, 0) << " " << pyramidBase(i, 1) << " " << pyramidBase(i, 2) << " 255 255 0\n"; // yellow base
        ++numVertices;
    }
    tempFilePyramid << pyramidApex.x() << " " << pyramidApex.y() << " " << pyramidApex.z() << " 255 255 0\n"; // yellow apex
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

void createAndSaveClassPointCloudVolumeData_multi_threads(const cv::Mat& classMatrix, const cv::Mat& tsdfMatrix, std::vector<Eigen::Matrix4f> poses, const std::string& outputFilename, Eigen::Vector3i volume_size, float voxel_scale, bool showVolumeCorners) {
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
        tempFilenames[i] = "classes_temp_" + std::to_string(i) + ".ply";
        int zStart = i * zStep;
        int zEnd = (i + 1) * zStep;
        if (i == numThreads - 1) zEnd = dz;
        threads[i] = std::thread(saveClassPointCloudProcessVolumeSlice, std::ref(classMatrix), std::ref(tsdfMatrix), tempFilenames[i], dx, dy, dz, zStart, zEnd, std::ref(numVerticesVec[i]), voxel_scale, showVolumeCorners);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (int nv : numVerticesVec) {
        numVertices += nv;
    }

    std::vector<std::string> camera_pose_tempFilenames;
    for (int i = 0; i < poses.size();) {
        // save the camera pose in the .ply file
        std::string camera_pose_tempFilename = "camera_pose_tempFile_" + std::to_string(i) + ".ply";
        camera_pose_tempFilenames.push_back(camera_pose_tempFilename);
        numVertices = save_camera_pose_point_cloud(poses[i], numVertices, camera_pose_tempFilename);
        i += 30; // save every 30 frames
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
        }
        else {
            std::cerr << "Error reading temporary file: " << tempFilename << std::endl;
        }
    }

    for (const auto& camera_pose_tempFilename : camera_pose_tempFilenames) {
        std::ifstream tempFilePyramid(camera_pose_tempFilename);
        if (tempFilePyramid.good()) {
            plyFile << tempFilePyramid.rdbuf();
            tempFilePyramid.close();
            std::remove(camera_pose_tempFilename.c_str());
        }
        else {
            std::cerr << "Error reading temporary file: " << camera_pose_tempFilename << std::endl;
        }
    }

    plyFile.close();
}

void saveClassPointCloudProcessVolumeSlice(const cv::Mat& classMatrix, const cv::Mat& tsdfMatrix, const std::string& tempFilename, int dx, int dy, int dz, int zStart, int zEnd, int& numVertices, float voxel_scale, bool showVolumeCorners) {
    std::ofstream tempFile(tempFilename);
    if (!tempFile.is_open()) {
        std::cerr << "Unable to open temporary file: " << tempFilename << std::endl;
        return;
    }

    for (int i = 0; i < dx; ++i) {
        for (int j = 0; j < dy; ++j) {
            for (int k = zStart; k < zEnd; ++k) {
                const int segmentLength = std::min({dx, dy, dz}) / 6;
                bool onVolumeCorners = (i <= segmentLength || (dx - 1 - i) <= segmentLength) &&
                                (j <= segmentLength || (dy - 1 - j) <= segmentLength) &&
                                (k <= segmentLength || (dz - 1 - k) <= segmentLength) &&
                                ((i == 0 && j == 0) || (i == 0 && k == 0) || (j == 0 && k == 0) || 
                                (i == dx - 1 && j == dy - 1) || (i == dx - 1 && k == dz - 1) || (j == dy - 1 && k == dz - 1) ||
                                (i == 0 && j == dy - 1) || (i == 0 && k == dz - 1) || (j == 0 && k == dz - 1) || 
                                (i == dx - 1 && j == 0) || (i == dx - 1 && k == 0) || (j == dy - 1 && k == 0));

                if (showVolumeCorners && onVolumeCorners){
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
                    float tsdfValue = tsdfMatrix.at<cv::Vec<short, 2>>(k * dy + j, i)[0] * DIVSHORTMAX;

                    if (abs(tsdfValue) < 0.2f * DIVSHORTMAX * SHORTMAX && tsdfValue != 0) {
                        // Retrieve the color value
                        uchar classValue =  classMatrix.at<uchar>(k * dy + j, i);
                        cv::Vec3b colorValue = getColorForClass(classValue);


                        if (colorValue == cv::Vec3b{ 0, 0, 0 }) {
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
    }
    tempFile.close();
}

void processNormalMapping(const cv::Mat& normal, const cv::Vec3f& lightPosition, const cv::Mat& vertex, cv::Mat& results, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int t = 0; t < normal.cols; t++) {
            const cv::Vec3f& vec = vertex.at<cv::Vec3f>(i, t);
            cv::Vec3f light = lightPosition - vec;  
            cv::normalize(light, light);

            const cv::Vec3f& nor = normal.at<cv::Vec3f>(i, t);
            float dotProduct = nor.dot(light);
            results.at<uchar>(i, t) = static_cast<uchar>((dotProduct + 1.0) * 0.5 * 255.0);
        }
    }
}

// L.N Shaded rendering
cv::Mat normalMapping(const cv::Mat& normal, const cv::Vec3f& lightPosition, const cv::Mat& vertex) {
    int row = normal.rows;
    int col = normal.cols;
    cv::Mat results(row, col, CV_8U);

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);

    int rowsPerThread = row / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == numThreads - 1) ? row : (i + 1) * rowsPerThread; // last thread handles remaining rows
        threads[i] = std::thread(processNormalMapping, std::cref(normal), lightPosition, std::cref(vertex), std::ref(results), startRow, endRow);
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    return results;
}

// Define the color palette
std::vector<std::vector<int>> ade20k_palette = {

		{120, 120, 120}, {180, 120, 120}, {6, 230, 230}, {80, 50, 50}, {4, 200, 3}, {120, 120, 80}, {140, 140, 140}, {204, 5, 255},
		{230, 230, 230}, {4, 250, 7}, {224, 5, 255}, {235, 255, 7}, {150, 5, 61}, {120, 120, 70}, {8, 255, 51}, {255, 6, 82},
		{143, 255, 140}, {204, 255, 4}, {255, 51, 7}, {204, 70, 3}, {0, 102, 200}, {61, 230, 250}, {255, 6, 51}, {11, 102, 255},
		{255, 7, 71}, {255, 9, 224}, {9, 7, 230}, {220, 220, 220}, {255, 9, 92}, {112, 9, 255}, {8, 255, 214}, {7, 255, 224},
		{255, 184, 6}, {10, 255, 71}, {255, 41, 10}, {7, 255, 255}, {224, 255, 8}, {102, 8, 255}, {255, 61, 6}, {255, 194, 7},
		{255, 122, 8}, {0, 255, 20}, {255, 8, 41}, {255, 5, 153}, {6, 51, 255}, {235, 12, 255}, {160, 150, 20}, {0, 163, 255},
		{140, 140, 140}, {250, 10, 15}, {20, 255, 0}, {31, 255, 0}, {255, 31, 0}, {255, 224, 0}, {153, 255, 0}, {0, 0, 255},
		{255, 71, 0}, {0, 235, 255}, {0, 173, 255}, {31, 0, 255}, {11, 200, 200}, {255, 82, 0}, {0, 255, 245}, {0, 61, 255},
		{0, 255, 112}, {0, 255, 133}, {255, 0, 0}, {255, 163, 0}, {255, 102, 0}, {194, 255, 0}, {0, 143, 255}, {51, 255, 0},
		{0, 82, 255}, {0, 255, 41}, {0, 255, 173}, {10, 0, 255}, {173, 255, 0}, {0, 255, 153}, {255, 92, 0}, {255, 0, 255},
		{255, 0, 245}, {255, 0, 102}, {255, 173, 0}, {255, 0, 20}, {255, 184, 184}, {0, 31, 255}, {0, 255, 61}, {0, 71, 255},
		{255, 0, 204}, {0, 255, 194}, {0, 255, 82}, {0, 10, 255}, {0, 112, 255}, {51, 0, 255}, {0, 194, 255}, {0, 122, 255},
		{0, 255, 163}, {255, 153, 0}, {0, 255, 10}, {255, 112, 0}, {143, 255, 0}, {82, 0, 255}, {163, 255, 0}, {255, 235, 0},
		{8, 184, 170}, {133, 0, 255}, {0, 255, 92}, {184, 0, 255}, {255, 0, 31}, {0, 184, 255}, {0, 214, 255}, {255, 0, 112},
		{92, 255, 0}, {0, 224, 255}, {112, 224, 255}, {70, 184, 160}, {163, 0, 255}, {153, 0, 255}, {71, 255, 0}, {255, 0, 163},
		{255, 204, 0}, {255, 0, 143}, {0, 255, 235}, {133, 255, 0}, {255, 0, 235}, {245, 0, 255}, {255, 0, 122}, {255, 245, 0},
		{10, 190, 212}, {214, 255, 0}, {0, 204, 255}, {20, 0, 255}, {255, 255, 0}, {0, 153, 255}, {0, 41, 255}, {0, 255, 204},
		{41, 0, 255}, {41, 255, 0}, {173, 0, 255}, {0, 245, 255}, {71, 0, 255}, {122, 0, 255}, {0, 255, 184}, {0, 92, 255},
		{184, 255, 0}, {0, 133, 255}, {255, 214, 0}, {25, 194, 194}, {102, 255, 0}, {92, 0, 255}

};

// Function to get the color for a given class index
cv::Vec3b getColorForClass(uchar classIndex) {
    int classIndexInt = static_cast<int>(classIndex);
	if (classIndexInt <= 0 || classIndexInt >= ade20k_palette.size()) {
		return cv::Vec3b(0, 0, 0);
	}
	return cv::Vec3b(ade20k_palette[classIndexInt][0], ade20k_palette[classIndexInt][1], ade20k_palette[classIndexInt][2]);

}

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}