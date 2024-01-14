#include "kinectfusion.hpp"
#include <iostream>
#define M_PI 3.14159265358979323846
Frame createCubeDepthMap(const Eigen::Matrix3f& intrinsics, const Eigen::Matrix4f& extrinsics, int width, int height, float cube_side, const Eigen::Vector3d& cube_center) {
    
    
    CameraParameters cameraParameters = CameraParameters(width, height, intrinsics);
    Frame syntheticFrame = Frame();
    syntheticFrame.camera_parameters = cameraParameters;
    syntheticFrame.extrinsic = extrinsics;


    Eigen::Matrix4f worldToCamera = extrinsics;
    Eigen::Matrix3f intrinsicsInv = intrinsics.inverse();

    // Intialize Depth Map
    syntheticFrame.depth_map = new float[width * height];
    syntheticFrame.color_map = new unsigned char[3 * width * height];

    std::cout << "worldToCamera: " << std::endl;
    std::cout << worldToCamera << std::endl;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Convert pixel coordinates to camera coordinates
            Eigen::Vector3f cameraDirection = (intrinsicsInv * Eigen::Vector3f(x, y, 1.0f)).normalized();

            // Convert direction to world coordinates
            Eigen::Vector3f worldDirection = worldToCamera.block<3, 3>(0, 0) * cameraDirection;

            // Flip the direction since the camera is looking in the negative z-direction
            worldDirection = -worldDirection;

            // Calculate depth as the distance to the cube
            float depth = (cube_center.cast<float>() - worldToCamera.block<3, 1>(0, 3)).dot(worldDirection);

            // Check if the point is inside the cube
            Eigen::Vector3d cube_min = cube_center - Eigen::Vector3d(cube_side / 2, cube_side / 2, cube_side / 2);
            Eigen::Vector3d cube_max = cube_center + Eigen::Vector3d(cube_side / 2, cube_side / 2, cube_side / 2);
            
            Eigen::Vector3f worldPoint = worldToCamera.block<3, 1>(0, 3) + depth * worldDirection;

            if (depth > 0 &&
                (worldPoint[0] >= cube_min[0] && worldPoint[0] <= cube_max[0]) &&
                (worldPoint[1] >= cube_min[1] && worldPoint[1] <= cube_max[1]) &&
                (worldPoint[2] >= cube_min[2] && worldPoint[2] <= cube_max[2])) {
               
                // Set depth value to the distance from the camera
                syntheticFrame.depth_map[y * width + x] = depth;

                // Set some color for visualization purposes
                syntheticFrame.color_map[3 * (y * width + x)] = 255; // Red channel
                syntheticFrame.color_map[3 * (y * width + x) + 1] = 255; // Green channel
                syntheticFrame.color_map[3 * (y * width + x) + 2] = 255; // Blue channel
            }
            else {
                // Set invalid depth
                syntheticFrame.depth_map[y * width + x] = -1.0f;
            }
        }
    }

    return syntheticFrame;
}

Frame createCuboidDepthMap(const Eigen::Matrix3f& intrinsics, const Eigen::Matrix4f& extrinsics, int width, int height, float cube_side, const Eigen::Vector3d& cube_center) {
    // The length of the cuboid along the x-axis is 2 * cube_side
    // The length of the cuboid along the y-axis is cube_side
    // The length of the cuboid along the z-axis is cube_side 



    CameraParameters cameraParameters = CameraParameters(width, height, intrinsics);
    Frame syntheticFrame = Frame();
    syntheticFrame.camera_parameters = cameraParameters;
    syntheticFrame.extrinsic = extrinsics;


    Eigen::Matrix4f worldToCamera = extrinsics;
    Eigen::Matrix3f intrinsicsInv = intrinsics.inverse();

    // Intialize Depth Map
    syntheticFrame.depth_map = new float[width * height];
    syntheticFrame.color_map = new unsigned char[3 * width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x++) {
			// Convert pixel coordinates to camera coordinates
			Eigen::Vector3f cameraDirection = (intrinsicsInv * Eigen::Vector3f(x, y, 1.0f)).normalized();

			// Convert direction to world coordinates
			Eigen::Vector3f worldDirection = worldToCamera.block<3, 3>(0, 0) * cameraDirection;

			// Calculate depth as the distance to the cubiod
            float depth = (cube_center.cast<float>() - worldToCamera.block<3, 1>(0, 3)).dot(worldDirection);

            // Check if the point is inside the cuboid
            Eigen::Vector3d cuboid_min = cube_center - Eigen::Vector3d(cube_side, cube_side / 2, cube_side / 2);
            Eigen::Vector3d cuboid_max = cube_center + Eigen::Vector3d(cube_side, cube_side / 2, cube_side / 2);

            Eigen::Vector3f worldPoint = worldToCamera.block<3, 1>(0, 3) + depth * worldDirection;

            if (depth > 0 &&
                (worldPoint[0] >= cuboid_min[0] && worldPoint[0] <= cuboid_max[0]) &&
                (worldPoint[1] >= cuboid_min[1] && worldPoint[1] <= cuboid_max[1]) &&
                (worldPoint[2] >= cuboid_min[2] && worldPoint[2] <= cuboid_max[2])) {

				// Set depth value to the distance from the camera
				syntheticFrame.depth_map[y * width + x] = depth;

				// Set some color for visualization purposes
				syntheticFrame.color_map[3 * (y * width + x)] = 255; // Red channel
				syntheticFrame.color_map[3 * (y * width + x) + 1] = 255; // Green channel
				syntheticFrame.color_map[3 * (y * width + x) + 2] = 255; // Blue channel
			}
            else {
				// Set invalid depth
				syntheticFrame.depth_map[y * width + x] = -1.0f;
			}

        }

    }
    return syntheticFrame;

}
int main(int argc, char** argv)
{
    int width = 640;
    int height = 480;
    Eigen::Matrix3f intrinsics;
    float focal_length = 500.0f; // adjust as needed
    float principal_point_x = width / 2.0f; // assuming the principal point is at the center
    float principal_point_y = height / 2.0f; // assuming the principal point is at the center

    intrinsics << focal_length, 0, principal_point_x,
        0, focal_length, principal_point_y,
        0, 0, 1;

    // Set up camera extrinsic matrix
    Eigen::Matrix4f extrinsics = Eigen::Matrix4f::Identity();
    Eigen::Vector3f pos = Eigen::Vector3f(0.0f, 0.0f, 2.0f); // Camera position
    
    extrinsics.block<3, 1>(0, 3) = pos;

    // Rotation matrix to look at the origin
    Eigen::Vector3d lookAtDirection = Eigen::Vector3d(0, 0,   2).normalized();

    // Choose an arbitrary up vector (can be optimized based on specific requirements)
    Eigen::Vector3d upVector(0, 1, 0);

    // Ensure the up vector is orthogonal to the look-at direction
    Eigen::Vector3d rightVector = upVector.cross(lookAtDirection).normalized();
    Eigen::Vector3d correctedUpVector = lookAtDirection.cross(rightVector);
    std::cout << correctedUpVector << std::endl;
    std::cout << lookAtDirection << std::endl;
    std::cout << rightVector << std::endl;

    // Rotation matrix to look at the origin
   
    Eigen::Matrix3d rotation;
    rotation.row(0) = rightVector;
    rotation.row(1) = correctedUpVector;
    rotation.row(2) =  lookAtDirection;
    
    
    extrinsics.block<3, 3>(0, 0) = rotation.cast<float>();

    std::cout << extrinsics << std::endl;


    float cube_side = 0.5f; // 50 cm
    Eigen::Vector3d cube_center(0.0, 0.0, 0.0);

    Frame syntheticFrame = createCubeDepthMap(intrinsics, extrinsics, width, height, cube_side, cube_center);

    // Save the depth map to disk using OpenCV
    cv::Mat depthMap(height, width, CV_32FC1, syntheticFrame.depth_map);

    // Scale and convert the depth values for saving as a grayscale image
    double minVal, maxVal;
    cv::minMaxLoc(depthMap, &minVal, &maxVal);
    std::cout << "Min depth: " << minVal << std::endl;
    std::cout << "Max depth: " << maxVal << std::endl;
    cv::Mat scaledDepthMap;
    depthMap.convertTo(scaledDepthMap, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // Save the scaled depth map
    cv::imwrite("depth_map.png", scaledDepthMap);

    int num_views = 10;

    // Loop to generate depth maps in a circular trajectory
    for (int view = 0; view < num_views; ++view) {
        // Calculate angle for this view
        float angle_rad = 2.0f * M_PI * view / num_views;

        // Set up camera extrinsic matrix for a circular trajectory with translation in the x-axis
        Eigen::Matrix4f extrinsics = Eigen::Matrix4f::Identity();
        extrinsics(0, 3) = 2.0f * cos(angle_rad);  // x-coordinate
        extrinsics(1, 3) = 2.0f * sin(angle_rad);  // y-coordinate
        extrinsics(2, 3) = 0.0f;                    // z-coordinate

        // Calculate the orientation of the camera to always look at the center of the cube
        Eigen::Vector3f cameraPosition = extrinsics.block<3, 1>(0, 3);
        Eigen::Vector3f cubeCenter = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f cameraDirection = -(cubeCenter -cameraPosition).normalized();
        Eigen::Vector3f up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        Eigen::Vector3f right = up.cross(cameraDirection).normalized();
        up = cameraDirection.cross(right).normalized();
        extrinsics.block<3, 1>(0, 0) = right;
        extrinsics.block<3, 1>(0, 1) = up;
        extrinsics.block<3, 1>(0, 2) = cameraDirection;

        // Create a synthetic depth map for the cube from this camera pose
        Frame syntheticFrame = createCubeDepthMap(intrinsics, extrinsics, width, height, cube_side, Eigen::Vector3d::Zero());

        // Save the depth map to disk using OpenCV
        cv::Mat depthMap(height, width, CV_32FC1, syntheticFrame.depth_map);
        double minVal, maxVal;
        cv::minMaxLoc(depthMap, &minVal, &maxVal);
        cv::Mat scaledDepthMap;
        depthMap.convertTo(scaledDepthMap, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        std::string filename = "depth_map_" + std::to_string(view) + ".png";
        cv::imwrite(filename, scaledDepthMap);
    }

}