#include "surface_measurement.hpp"
#include <opencv2/opencv.hpp>
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
    // create vertex and normal pyramid
    for (size_t i = 0; i < num_levels; i++)
    {
        compute_map(data.smoothed_depth_pyramid[i], camera_params.level(i), data.vertex_pyramid[i], data.normal_pyramid[i],depth_cutoff);
    }
    return data;


}

void compute_map(const cv::Mat_<float>& depthmap, const CameraParameters& camera_params, cv::Mat & vertexMap, cv::Mat & normalMap, const float & depth_cutoff)

{
    int row = depthmap.rows;
    int col = depthmap.cols;
    // compute vertex 
    for (size_t i = 0; i < row; i++)
    {
        for (size_t t = 0; t < col; t++)
        {   
            float depth= depthmap(i,t);
            cv::Vec3f& vertex = vertexMap.at<cv::Vec3f>(i, t);
            if (depth>depth_cutoff)
            {
                depth=0.f;
            }
            // Setting X, Y, and Z components of the 3D vecto
            vertex[0] = (t - camera_params.principal_x) * depth / camera_params.focal_x;
            vertex[1] = (i - camera_params.principal_y) * depth / camera_params.focal_y;
            vertex[2] = depth;

        }
    }
    // compute normal
    for (size_t i = 0; i < row; i++)
    {
        
        for (size_t t = 0; t < col; t++)
        {   
            if (i==0||t==0||i==row-1||t==col-1)
            {
               normalMap.at<cv::Vec3f>(i, t)=cv::Vec3f(0,0,0);
            }
            else
            {
                cv::Vec3f p0 = vertexMap.at<cv::Vec3f>(i-1, t);
                cv::Vec3f p1 = vertexMap.at<cv::Vec3f>(i+1, t);
                cv::Vec3f p2 = vertexMap.at<cv::Vec3f>(i , t-1);
                cv::Vec3f p3 = vertexMap.at<cv::Vec3f>(i , t + 1);
                cv::Vec3f dp_dx = (p1 - p0) ;
                cv::Vec3f dp_dy = (p2 - p3 );

                
                cv::Vec3f normal = dp_dx.cross(dp_dy);
                if (cv::norm(normal) != 0.0)
                {
                    cv::normalize(normal,normal,1);
                }
                
                normalMap.at<cv::Vec3f>(i, t) = normal;
            }
        }
    }

}
   
