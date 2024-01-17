#include "surface_prediction.hpp"

#define DIVSHORTMAX 0.0000305185f 
#define SHORTMAX    32767               // SHRT_MAX;
#define MAX_WEIGHT  128                // max weight

// tri-linear interpolation
float interpolate_trilinearly(
    Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& point,
    cv::Mat& volume,  // TSDF Volume
    const Eigen::Vector3i& volume_size,
    const float voxel_scale)
{
    //cooridnate in grid and in int
    Eigen::Matrix<int, 3, 1, Eigen::DontAlign> point_in_grid = point.cast<int>();

    // voxel center
    const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
    const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
    const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);
       
    point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
    point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
    point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

    // ref: https://en.wikipedia.org/wiki/Trilinear_interpolation
    const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
    const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
    const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

    return 
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z()) * volume_size[1] + point_in_grid.y()),(point_in_grid.x()))[0]) * DIVSHORTMAX 
            // volume[ x ][ y ][ z ], C000
            * (1 - a) * (1 - b) * (1 - c) +
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z() + 1) * volume_size[1] + point_in_grid.y()),(point_in_grid.x()))[0]) * DIVSHORTMAX 
            // volume[ x ][ y ][z+1], C001
            * (1 - a) * (1 - b) * c +
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z()) * volume_size[1] + point_in_grid.y() + 1),(point_in_grid.x()))[0]) * DIVSHORTMAX 
            // volume[ x ][y+1][ z ], C010
            * (1 - a) * b * (1 - c) +
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z() + 1) * volume_size[1] + point_in_grid.y() + 1),(point_in_grid.x()))[0]) * DIVSHORTMAX 
            // volume[ x ][y+1][z+1], C011
            * (1 - a) * b * c +
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z()) * volume_size[1] + point_in_grid.y()),(point_in_grid.x() + 1))[0]) * DIVSHORTMAX 
            // volume[x+1][ y ][ z ], C100
            * a * (1 - b) * (1 - c) +
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z() + 1) * volume_size[1] + point_in_grid.y()),(point_in_grid.x() + 1))[0]) * DIVSHORTMAX 
            // volume[x+1][ y ][z+1], C101
            * a * (1 - b) * c +
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z()) * volume_size[1] + point_in_grid.y() + 1),(point_in_grid.x() + 1))[0]) * DIVSHORTMAX 
            // volume[x+1][y+1][ z ], C110
            * a * b * (1 - c) +
        static_cast<float>(volume.at<cv::Vec<short, 2>>(static_cast<int>((point_in_grid.z() + 1) * volume_size[1] + point_in_grid.y() + 1),(point_in_grid.x() + 1))[0]) * DIVSHORTMAX 
            // volume[x+1][y+1][z+1], C111
            * a * b * c;
}

float get_min_time(
    const Eigen::Vector3f& volume_max,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& origin,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& direction)
{
    float txmin = ((direction.x() > 0 ? 0.f : volume_max[0]) - origin.x()) / direction.x();
    // std::cout << "volume_max[0] " << volume_max[0] << std::endl;
    // std::cout << "direction.x " << direction.x() << std::endl;
    // std::cout << "origin.x " << origin.x() << std::endl;
    // std::cout << "txmin " << txmin << std::endl;
    // std::cout << "sxmin " << txmin * direction.x() << std::endl;

    float tymin = ((direction.y() > 0 ? 0.f : volume_max[1]) - origin.y()) / direction.y();
    // std::cout << "direction.y " << direction.y() << std::endl;
    // std::cout << "origin.y " << origin.y() << std::endl;
    // std::cout << "tymin " << tymin << std::endl;
    // std::cout << "symin " << tymin * direction.y() << std::endl;

    float tzmin = ((direction.z() > 0 ? 0.f : volume_max[2]) - origin.z()) / direction.z();
    // std::cout << "direction.z " << direction.z() << std::endl;
    // std::cout << "origin.z " << origin.z() << std::endl;
    // std::cout << "tzmin " << tzmin << std::endl;
    // std::cout << "szmin " << tzmin * direction.z() << std::endl;
    // std::cout << std::endl;
    // std:;cout << "txmin " << txmin << std::endl;
    // std::cout << "tymin " << tymin << std::endl;
    // std::cout << "tzmin " << tzmin << std::endl;

    return fmax(fmax(txmin, tymin), tzmin);
}

float get_max_time(
    const Eigen::Vector3f& volume_max,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& origin,
    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& direction
)
{
    float txmax = ((direction.x() > 0 ? volume_max[0] : 0.f) - origin.x()) / direction.x();
    float tymax = ((direction.y() > 0 ? volume_max[1] : 0.f) - origin.y()) / direction.y();
    float tzmax = ((direction.z() > 0 ? volume_max[2] : 0.f) - origin.z()) / direction.z();
    // std::cout << "txmax " << txmax << std::endl;
    // std::cout << "tymax " << tymax << std::endl;
    // std::cout << "tzmax " << tzmax << std::endl;

    return fmin(fmin(txmax, tymax), tzmax);
}

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
)
{
    // std::cout << "\nmodel_vertex size: " << model_vertex.size() << std::endl;
    for(int x = 0; x < model_vertex.cols; ++x){
        for(int y = 0; y < model_vertex.rows; ++y){
    // for(int x = model_vertex.cols/2.f; x < model_vertex.cols; ++x){
    //     for(int y = model_vertex.rows/2.f; y < model_vertex.rows; ++y){
            // std::cout << "(x,y) = (" << x << "," << y << ")" << std::endl;
            const Eigen::Vector3f volume_range = volume_size.cast<float>() * voxel_scale; 
            const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> pixel_position(
                        (x - cam_parameters.principal_x) / cam_parameters.focal_x,      // X/Z
                        (y - cam_parameters.principal_y) / cam_parameters.focal_y,      // Y/Z
                        1.f);
            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> ray_direction = rotation * pixel_position;
            ray_direction.normalize();

            // std::cout << "ray_direction: \n" << ray_direction << std::endl;

            float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
            // std::cout << "ray_length / get_min_time: " << ray_length << std::endl;
            float max_time = get_max_time(volume_range, translation, ray_direction);
            // std::cout << "get_max_time: " << max_time << std::endl;

            if (ray_length >= max_time){
                // std::cout << "ray_length >= get_max_time" << std::endl;
                continue;
            }
            
            ray_length += voxel_scale;
            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> grid = (translation + (ray_direction * ray_length)) / voxel_scale;
            // std::cout << "grid: " << grid << std::endl;

            float tsdf = static_cast<float>(tsdf_volume.at<cv::Vec<short, 2>>(
                (static_cast<int>(std::floor(grid(2)) * volume_size[1] + std::floor(grid(1)))),
                (static_cast<int>(std::floor(grid(0)))))[0]) * DIVSHORTMAX;

            // std::cout << "tsdf: " << tsdf << std::endl;
            
            //TODO: why not sqrt(3) * volume_range[0]?
            const float max_search_length = ray_length + volume_range[0] * sqrt(2.f);

            for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f){

                grid = ((translation + (ray_direction * (ray_length + truncation_distance * 0.5f))) / voxel_scale);
                
                //checkvalid
                if (grid.x() < 1 || grid.x() >= volume_size[0] - 1 || 
                    grid.y() < 1 || grid.y() >= volume_size[1] - 1 ||
                    grid.z() < 1 || grid.z() >= volume_size[2] - 1 )
                        continue;
                
                const float previous_tsdf = tsdf;

                float tsdf = static_cast<float>(tsdf_volume.at<cv::Vec<short, 2>>(
                (static_cast<int>(std::floor(grid(2)) * volume_size[1] + std::floor(grid(1)))),
                (static_cast<int>(std::floor(grid(0)))))[0]) * DIVSHORTMAX;

                // if (tsdf != 0.f){
                //     std::cout << "(tsdf, weight) = (" << tsdf << ", " << static_cast<float>(tsdf_volume.at<cv::Vec<short, 2>>(
                //         (static_cast<int>(std::floor(grid(2)) * volume_size[1] + std::floor(grid(1)))),
                //         (static_cast<int>(std::floor(grid(0)))))[1]) << ")" << std::endl;
                // }

                // if ray enter from behind
                if (previous_tsdf < 0.f && tsdf > 0.f)
                    break;

                //if intersecting zero-crossing
                if (previous_tsdf > 0.f && tsdf < 0.f){

                    const float t_star =
                                ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
                    
                    const auto vertex = translation + ray_direction * t_star;

                    const Eigen::Matrix<float, 3, 1, Eigen::DontAlign> location_in_grid = (vertex / voxel_scale);

                    // check if location is in the grid
                    if (location_in_grid.x() < 1 || location_in_grid.x() >= volume_size[0] - 1 ||
                            location_in_grid.y() < 1 || location_in_grid.y() >= volume_size[1] - 1 ||
                            location_in_grid.z() < 1 || location_in_grid.z() >= volume_size[2] - 1)
                            break;

                    // caculate normal in this position

                    Eigen::Matrix<float, 3, 1, Eigen::DontAlign> normal;
                    Eigen::Matrix<float, 3, 1, Eigen::DontAlign> shifted;

                    // x direction
                    shifted = location_in_grid;

                    shifted.x() += 1.f;

                    if(shifted.x() >= volume_size[0] - 1)
                        break;
                    
                    const float Fx1 = interpolate_trilinearly(
                        shifted, 
                        tsdf_volume, 
                        volume_size, 
                        voxel_scale);

                    // another shifted direction
                    shifted = location_in_grid;

                    shifted.x() -= 1.f;

                    if(shifted.x() < 1)
                        break;

                    const float Fx2 = interpolate_trilinearly(
                        shifted, 
                        tsdf_volume, 
                        volume_size, 
                        voxel_scale);

                    normal.x() = (Fx1 - Fx2);

                    // y direction
                    shifted = location_in_grid;

                    shifted.y() += 1;

                    if (shifted.y() >= volume_size[1] - 1)
                        break;

                    const float Fy1 = interpolate_trilinearly(
                        shifted, 
                        tsdf_volume, 
                        volume_size, 
                        voxel_scale);

                    shifted = location_in_grid;

                    shifted.y() -= 1;

                    if (shifted.y() < 1)
                        break;

                    const float Fy2 = interpolate_trilinearly(
                        shifted, 
                        tsdf_volume, 
                        volume_size, 
                        voxel_scale);

                    normal.y() = (Fy1 - Fy2);

                    // z direction
                    shifted = location_in_grid;

                    shifted.z() += 1;

                    if (shifted.z() >= volume_size[2] - 1)
                        break;

                    const float Fz1 = interpolate_trilinearly(
                        shifted, 
                        tsdf_volume, 
                        volume_size, 
                        voxel_scale);

                    shifted = location_in_grid;

                    shifted.z() -= 1;

                    if (shifted.z() < 1)
                        break;

                    const float Fz2 = interpolate_trilinearly(
                        shifted, 
                        tsdf_volume, 
                        volume_size, 
                        voxel_scale);

                    normal.z() = (Fz1 - Fz2);

                    if (normal.norm() == 0)
                            break;

                    //if normal is not zero, then normalize it
                    normal.normalize();

                    //save vertex, normal
                    model_vertex.at<cv::Vec3f>(y,x) = cv::Vec3f(vertex.x(), vertex.y(), vertex.z());
                    model_normal.at<cv::Vec3f>(y,x) = cv::Vec3f(normal.x(), normal.y(), normal.z());

                    //save color
                    auto location_in_grid_int = location_in_grid.cast<int>();

                    model_color.at<cv::Vec3b>(y,x) = color_volume.at<cv::Vec3b>((
                                location_in_grid_int.z() * volume_size[1] +
                                location_in_grid_int.y()),(location_in_grid_int.x()));
                    
                    break;
                }
            }
        }
    }
}

void surface_prediction(
    VolumeData& volume,                   // Global Volume
    cv::Mat& model_vertex,                       // predicted vertex
    cv::Mat& model_normal,                       // predicted normal
    cv::Mat& model_color,                        // predicted color
    const CameraParameters& cam_parameters,     
    const float truncation_distance,            
    const Eigen::Matrix4f& pose)                // camera pose
{
    // data praparation, empty the model_vertex, model_normal, model_color
    model_vertex.setTo(0);
    model_normal.setTo(0);
    model_color.setTo(0);

    raycast_tsdf_kernel(
        volume.tsdf_volume,                 // Global TSDF Volume
        volume.color_volume,                // Global Color Volume
        model_vertex,                       // predicted vertex
        model_normal,                       // predicted normal
        model_color,                        // predicted color
        volume.volume_size,                 
        volume.voxel_scale,                 
        cam_parameters,                     
        truncation_distance,                
        pose.block(0, 0, 3, 3),             // rotation matrix
        pose.block(0, 3, 3, 1));            // translation vector
    // std::cout << ">> 4.* raycast_tsdf_kernel done" << std::endl;
}