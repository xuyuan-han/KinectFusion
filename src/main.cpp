#include "kinectfusion.hpp"
#include <iostream>
#define M_PI 3.14159265358979323846


bool notintersectCube(const Eigen::Vector3d& cube_center, float cube_side, const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction, float& t) {
    // Compute half extents along each axis
    Eigen::Vector3d cube_min = cube_center - Eigen::Vector3d(cube_side / 2, cube_side / 2, cube_side / 2);
    Eigen::Vector3d cube_max = cube_center + Eigen::Vector3d(cube_side / 2, cube_side / 2, cube_side / 2);

    // Compute the intersection of the ray with the cube
    double tmin, tmax, tymin, tymax, tzmin, tzmax;

    // Calculate inverse direction components
    Eigen::Vector3d ray_direction_inv = Eigen::Vector3d(1.0f / ray_direction[0], 1.0f / ray_direction[1], 1.0f / ray_direction[2]);

    // X slab
    tmin = (cube_min[0] - ray_origin[0]) * ray_direction_inv[0];
    tmax = (cube_max[0] - ray_origin[0]) * ray_direction_inv[0];

    // Y slab
    tymin = (cube_min[1] - ray_origin[1]) * ray_direction_inv[1];
    tymax = (cube_max[1] - ray_origin[1]) * ray_direction_inv[1];

    if ((tmin > tymax) || (tymin > tmax)) {
        return false;
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    // Z slab
    tzmin = (cube_min[2] - ray_origin[2]) * ray_direction_inv[2];
    tzmax = (cube_max[2] - ray_origin[2]) * ray_direction_inv[2];

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false;
    }

    if (tzmin > tmin) {
        tmin = tzmin;
    }

    if (tzmax < tmax) {
        tmax = tzmax;
    }

    // The ray intersects the cube if tmin is less than or equal to tmax
    // and tmax is greater than or equal to 0
    return (tmin <= tmax && tmax >= 0);
}
#include <Eigen/Dense>

bool intersectCube(const Eigen::Vector3d& cube_center, float cube_side, const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction, float& t) {
    // Compute half extents along each axis
    Eigen::Vector3d cube_min = cube_center - Eigen::Vector3d(cube_side / 2, cube_side / 2, cube_side / 2);
    Eigen::Vector3d cube_max = cube_center + Eigen::Vector3d(cube_side / 2, cube_side / 2, cube_side / 2);

    // Initialize t to a very large value
    t = std::numeric_limits<float>::max();

    // Check each face of the cube
    for (int i = 0; i < 3; ++i) {
        // Check min face
        float t_min = (cube_min[i] - ray_origin[i]) / ray_direction[i];
        if (t_min < t) {
            Eigen::Vector3f intersection_point = ray_origin + t_min * ray_direction;
            if (intersection_point[0] >= cube_min[0] && intersection_point[0] <= cube_max[0] &&
                intersection_point[1] >= cube_min[1] && intersection_point[1] <= cube_max[1] &&
                intersection_point[2] >= cube_min[2] && intersection_point[2] <= cube_max[2]) {
                t = t_min;
            }
        }

        // Check max face
        float t_max = (cube_max[i] - ray_origin[i]) / ray_direction[i];
        if (t_max < t) {
            Eigen::Vector3f intersection_point = ray_origin + t_max * ray_direction;
            if (intersection_point[0] >= cube_min[0] && intersection_point[0] <= cube_max[0] &&
                intersection_point[1] >= cube_min[1] && intersection_point[1] <= cube_max[1] &&
                intersection_point[2] >= cube_min[2] && intersection_point[2] <= cube_max[2]) {
                t = t_max;
            }
        }
    }

    // Check if the ray intersects the cube
    bool hit = t < std::numeric_limits<float>::max();

    return hit;
}

struct Point {
    float x, y, z;
    uint8_t r, g, b; // Color
};

void createAndSavePointCloud(const cv::Mat& tsdfMatrix, const std::string& outputFilename) {
    std::ofstream plyFile(outputFilename);

    if (!plyFile.is_open()) {
        std::cerr << "Unable to open file: " << outputFilename << std::endl;
        return;
    }

    // Write to something temprary and then copy to the final file
    // This is done to update the number of vertices in the header
    std::ofstream tempFile("temp.ply");
    // Keep track of the number of vertices
    int numVertices = 0;

    for (int x = 0; x < tsdfMatrix.size[0]; ++x) {
        for (int y = 0; y < tsdfMatrix.size[1]; ++y) {
            for (int z = 0; z < tsdfMatrix.size[2]; ++z) {
                Point point;

                // Extract TSDF value
                float tsdfValue = tsdfMatrix.at<float>(x, y, z);
                if (abs(tsdfValue) > 0.02f || tsdfValue == 0) {
                    // Skip TSDF value if it is invalid
                    continue;
                }

                // Set point coordinates
                point.x = static_cast<float>(x);
                point.y = static_cast<float>(y);
                point.z = static_cast<float>(z);

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

    // Close the temporary file
    tempFile.close();
    
    // Write the header to the final file
    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << numVertices << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "property uchar red\n";
    plyFile << "property uchar green\n";
    plyFile << "property uchar blue\n";
    plyFile << "end_header\n";

    // Copy the contents of the temporary file to the final file
    std::ifstream tempFileRead("temp.ply");
    plyFile << tempFileRead.rdbuf();
    tempFileRead.close();

    // Close the PLY file
    plyFile.close();

    std::cout << "Point cloud saved to " << outputFilename << std::endl;
}




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
    syntheticFrame.class_map = new unsigned int[width * height];


    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Convert pixel coordinates to camera coordinates
            Eigen::Vector3f cameraDirection = (intrinsicsInv * Eigen::Vector3f(x, y, 1.0f)).normalized();

            cameraDirection[1] = -cameraDirection[1];
     
            // Convert direction to world coordinates
            Eigen::Vector3f worldDirection = worldToCamera.block<3, 3>(0, 0) * cameraDirection;


            // Flip the direction since the camera is looking in the negative z-direction
            worldDirection = worldDirection.normalized();
            //std::cout << "worldDirection: " << worldDirection << std::endl;
            //std::cout << "worldToCamera.block<3, 1>(0, 3): " << worldToCamera.block<3, 1>(0, 3) << std::endl;

            // Find intersection point with the cube
           
            float t;
            bool hit = intersectCube(cube_center, cube_side, worldToCamera.block<3, 1>(0, 3), worldDirection, t);
           
            if (hit) {
                // Calculate depth as the distance to the intersection point
                float depth = t;
                
                // We knwo have the depth along the ray, but we need to find the orthogonal distance to the camera
                // We can do this by projecting the intersection point onto the optical axis
                Eigen::Vector3f opticalAxis = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
                opticalAxis = worldToCamera.block<3, 3>(0, 0) * opticalAxis;

                depth = (depth * worldDirection).dot(opticalAxis);


                // Set depth value to the distance from the camera
                syntheticFrame.depth_map[y * width + x] = depth;

                // Set some color for visualization purposes
                syntheticFrame.color_map[3 * (y * width + x)] = 255; // Red channel
                syntheticFrame.color_map[3 * (y * width + x) + 1] = 255; // Green channel
                syntheticFrame.color_map[3 * (y * width + x) + 2] = 255; // Blue channel
            }
            else {
                // Set invalid depth
                syntheticFrame.depth_map[y * width + x] = -1;
            }

            syntheticFrame.class_map[y * width + x] = 0;
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
            
            worldDirection = -worldDirection.normalized();
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
    float focal_length = 300; // adjust as needed
    float principal_point_x = width / 2.0f; // assuming the principal point is at the center
    float principal_point_y = height / 2.0f; // assuming the principal point is at the center

    intrinsics << focal_length, 0, principal_point_x,
        0, focal_length, principal_point_y,
        0, 0, 1;

    float cube_side = 1.0f; // 0 cm
    Eigen::Vector3d cube_center(0.0, 0.0, 0.0);

    // Setup  Volume(Eigen::Vector3d min_, Eigen::Vector3d max_, uint dx_, uint dy_, uint dz_, uint dim)
    unsigned int mc_res = 100; // resolution of the grid, for debugging you can reduce the resolution (-> faster)
    Volume vol(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), mc_res, mc_res, mc_res, 1);


    int num_views = 4;
    // Loop to generate depth maps in a circular trajectory
    for (int view = 0; view < num_views; ++view) {

        

        // Calculate angle for this view
        float angle_rad = 2.0f * M_PI * view / num_views;

        // Set up camera extrinsic matrix for a circular trajectory with translation in the x-axis
        Eigen::Matrix4f extrinsics = Eigen::Matrix4f::Identity();
        extrinsics(0, 3) = 3* cos(angle_rad);  // x-coordinate
        extrinsics(1, 3) = 3* sin(angle_rad);  // y-coordinate
        extrinsics(2, 3) = 0.0f;                    // z-coordinate

        // Calculate the orientation of the camera to always look at the center of the cube
        Eigen::Vector3f cameraPosition = extrinsics.block<3, 1>(0, 3);
        Eigen::Vector3f cubeCenter = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f cameraDirection = (cubeCenter - cameraPosition).normalized();
        Eigen::Vector3f up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);

        // Make up vector orthogonal to camera direction
        // Corrected cross product order
        Eigen::Vector3f right = cameraDirection.cross(up).normalized();
        up = right.cross(cameraDirection).normalized();
        extrinsics.block<3, 1>(0, 0) = right;
        extrinsics.block<3, 1>(0, 1) = up;
        extrinsics.block<3, 1>(0, 2) = cameraDirection;

        // Create a synthetic depth map for the cube from this camera pose
        Frame syntheticFrame = createCubeDepthMap(intrinsics, extrinsics, width, height, cube_side, Eigen::Vector3d::Zero());

        //! Save the depth map to disk using OpenCV
       cv::Mat depthMap(height, width, CV_32FC1, syntheticFrame.depth_map);
       double minVal, maxVal;
       cv::minMaxLoc(depthMap, &minVal, &maxVal);
       cv::Mat scaledDepthMap;
       depthMap.convertTo(scaledDepthMap, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
       std::string filename = "depth_map_" + std::to_string(view) + ".png";
       cv::imwrite(filename, scaledDepthMap);

        Surface_Reconstruction::integrate(&vol, syntheticFrame, 0.02f);
    
        //Save the volume to disk as a point cloud with the name point_cloud_0.ply, point_cloud_1.ply, etc.
        createAndSavePointCloud(vol.getVolume(), "point_cloud_" + std::to_string(view) + ".ply");
    }


    // ! Now 10 frames on yz plane

    


}