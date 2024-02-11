#include "surface_measurement.hpp"
FrameData surface_measurement(const cv::Mat_<float>& input_frame,
    const CameraParameters& camera_params,
    const size_t num_levels, const float depth_cutoff,
    const int kernel_size, const float color_sigma, const float spatial_sigma)
{
    // Initialize frame data
    FrameData data(num_levels);
    for (size_t level = 0; level < num_levels; ++level) 
    {
    const int width = camera_params.level(level).image_width;
    const int height = camera_params.level(level).image_height;

    // Create a regular cv::Mat for the depth pyramid at the current level
    data.depth_pyramid[level] = cv::Mat(height, width, CV_32FC1);

    // Create a regular cv::Mat for the smoothed depth pyramid at the current level
    data.smoothed_depth_pyramid[level] = cv::Mat(height, width, CV_32FC1);

    // Create a regular cv::Mat for the color pyramid at the current level
    data.color_pyramid[level] = cv::Mat(height, width, CV_8UC3);

    // Create a regular cv::Mat for the vertex pyramid at the current level
    data.vertex_pyramid[level] = cv::Mat(height, width, CV_32FC3);

    // Create a regular cv::Mat for the normal pyramid at the current level
    data.normal_pyramid[level] = cv::Mat(height, width, CV_32FC3);
    }
    
    int row = input_frame.rows;
    int col = input_frame.cols;
    data.depth_pyramid[0]=input_frame;
    // Initialize depth cut off
    // create depth pyramid
   
    for (size_t level = 1; level < num_levels; ++level)
        cv::pyrDown(data.depth_pyramid[level - 1], data.depth_pyramid[level]);
    // create smooth pyramid
    for (size_t level = 0; level < num_levels; ++level) {
        cv::bilateralFilter(data.depth_pyramid[level], // source
            data.smoothed_depth_pyramid[level], // destination
            kernel_size,
            color_sigma,
            spatial_sigma,
            cv::BORDER_DEFAULT
            );
    }
    #ifdef USE_CPU_MULTI_THREADING
    for (size_t i = 0; i < num_levels; i++)
    {
        compute_map_multi_threads(data.smoothed_depth_pyramid[i], camera_params.level(i), data.vertex_pyramid[i], data.normal_pyramid[i], depth_cutoff);
    }
    #else
    for (size_t i = 0; i < num_levels; i++)
    {
        compute_map(data.smoothed_depth_pyramid[i], camera_params.level(i), data.vertex_pyramid[i], data.normal_pyramid[i], depth_cutoff);
    }
    #endif
    return data;

}

void compute_map(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat & vertexMap, cv::Mat & normalMap, const float & depth_cutoff)

{
    int row = depthmap.rows;
    int col = depthmap.cols; 
    compute_vertex_map(depthmap, camera_params, vertexMap, depth_cutoff, 0, row);
    
    compute_normal_map(depthmap,vertexMap,normalMap,0,row);

}

void compute_map_multi_threads(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat& vertexMap, cv::Mat& normalMap, const float& depth_cutoff)
{
    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    int rowsPerThread = depthmap.rows / numThreads;
    int start_row = 0;
    int end_row = 0;
    for (int i = 0; i < numThreads; ++i)
    {
        end_row = (i == numThreads - 1) ? depthmap.rows : start_row + rowsPerThread;
        threads[i] = std::thread(compute_vertex_map, std::ref(depthmap), std::ref(camera_params), std::ref(vertexMap), std::ref(depth_cutoff), start_row, end_row);
        start_row = end_row;
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
    std::vector<std::thread> normalThreads(numThreads);
    start_row = 0;
    for (int i = 0; i < numThreads; ++i)
    {
        end_row = (i == numThreads - 1) ? depthmap.rows : start_row + rowsPerThread;
        normalThreads[i] = std::thread(compute_normal_map, std::ref(depthmap), std::ref(vertexMap), std::ref(normalMap), start_row, end_row);
        start_row = end_row;
    }

    for (auto& thread : normalThreads)
    {
        thread.join();
    }
}

void compute_vertex_map(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat& vertexMap, const float& depth_cutoff, int start_row, int end_row)
{
    int col = depthmap.cols;
    for (size_t i = start_row; i < end_row; i++)
    {
        for (size_t t = 0; t < col; t++)
        {
            float depth = depthmap(i, t);
            cv::Vec3f& vertex = vertexMap.at<cv::Vec3f>(i, t);
            if (depth > depth_cutoff)
            {
                depth = 0.f;
            }
            vertex[0] = (t - camera_params.principal_x) * depth / camera_params.focal_x;
            vertex[1] = (i - camera_params.principal_y) * depth / camera_params.focal_y;
            vertex[2] = depth;
        }
    }
}

void compute_normal_map(const cv::Mat_<float>& depthmap, const cv::Mat& vertexMap, cv::Mat& normalMap, int start_row, int end_row)
{
    int col = depthmap.cols;
    for (size_t i = start_row; i < end_row; i++)
    {
        for (size_t t = 0; t < col; t++)
        {
            if (i == 0 || t == 0 || i == depthmap.rows - 1 || t == col - 1)
            {
                normalMap.at<cv::Vec3f>(i, t) = cv::Vec3f(0, 0, 0);
            }
            else
            {
                cv::Vec3f p0 = vertexMap.at<cv::Vec3f>(i - 1, t); // upper
                cv::Vec3f p1 = vertexMap.at<cv::Vec3f>(i + 1, t); // lower
                cv::Vec3f p2 = vertexMap.at<cv::Vec3f>(i, t - 1); // left
                cv::Vec3f p3 = vertexMap.at<cv::Vec3f>(i, t + 1); // right
                cv::Vec3f dp_dy = (p0 - p1); // lower -> upper
                cv::Vec3f minus_dp_dx = (p2 - p3); // right -> left

                cv::Vec3f normal = dp_dy.cross(minus_dp_dx);

                if (normal[2] > 0) {
                    normal *= -1;
                }

                if (cv::norm(normal) != 0.0)
                {
                    cv::normalize(normal, normal, 1);
                }

                normalMap.at<cv::Vec3f>(i, t) = normal;
            }
        }
    }
}