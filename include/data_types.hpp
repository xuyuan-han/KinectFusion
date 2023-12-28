#pragma once

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>

struct CameraParameters {
    int image_width, image_height;
    float focal_x, focal_y;
    float principal_x, principal_y;

    /**
     * Returns camera parameters for a specified pyramid level; each level corresponds to a scaling of pow(.5, level)
     * @param level The pyramid level to get the parameters for with 0 being the non-scaled version,
     * higher levels correspond to smaller spatial size
     * @return A CameraParameters structure containing the scaled values
     */
    CameraParameters level(const size_t level) const
    {
        if (level == 0) return *this;

        const float scale_factor = powf(0.5f, static_cast<float>(level));
        return CameraParameters { image_width >> level, image_height >> level,
                                    focal_x * scale_factor, focal_y * scale_factor,
                                    (principal_x + 0.5f) * scale_factor - 0.5f,
                                    (principal_y + 0.5f) * scale_factor - 0.5f };
    }
};

struct FrameData {
    std::vector<cv::Mat> depth_pyramid;
    std::vector<cv::Mat> smoothed_depth_pyramid;
    std::vector<cv::Mat> color_pyramid;

    std::vector<cv::Mat> vertex_pyramid;
    std::vector<cv::Mat> normal_pyramid;

    explicit FrameData(const size_t pyramid_height) :
            depth_pyramid(pyramid_height), smoothed_depth_pyramid(pyramid_height),
            color_pyramid(pyramid_height), vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height)
    { }

    // No copying
    FrameData(const FrameData&) = delete;
    FrameData& operator=(const FrameData& other) = delete;

    FrameData(FrameData&& data) noexcept :
            depth_pyramid(std::move(data.depth_pyramid)),
            smoothed_depth_pyramid(std::move(data.smoothed_depth_pyramid)),
            color_pyramid(std::move(data.color_pyramid)),
            vertex_pyramid(std::move(data.vertex_pyramid)),
            normal_pyramid(std::move(data.normal_pyramid))
    { }

    FrameData& operator=(FrameData&& data) noexcept
    {
        depth_pyramid = std::move(data.depth_pyramid);
        smoothed_depth_pyramid = std::move(data.smoothed_depth_pyramid);
        color_pyramid = std::move(data.color_pyramid);
        vertex_pyramid = std::move(data.vertex_pyramid);
        normal_pyramid = std::move(data.normal_pyramid);
        return *this;
    }
};

struct ModelData {
    std::vector<cv::Mat> vertex_pyramid;
    std::vector<cv::Mat> normal_pyramid;
    std::vector<cv::Mat> color_pyramid;

    ModelData(const size_t pyramid_height, const CameraParameters camera_parameters) :
            vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height),
            color_pyramid(pyramid_height)
    {
        for (size_t level = 0; level < pyramid_height; ++level) {
            vertex_pyramid[level] =
                    cv::Mat(camera_parameters.level(level).image_height,
                                                camera_parameters.level(level).image_width,
                                                CV_32FC3);
            normal_pyramid[level] =
                    cv::Mat(camera_parameters.level(level).image_height,
                                                camera_parameters.level(level).image_width,
                                                CV_32FC3);
            color_pyramid[level] =
                    cv::Mat(camera_parameters.level(level).image_height,
                                                camera_parameters.level(level).image_width,
                                                CV_8UC3);
            vertex_pyramid[level].setTo(0);
            normal_pyramid[level].setTo(0);
        }
    }

    // No copying
    ModelData(const ModelData&) = delete;
    ModelData& operator=(const ModelData& data) = delete;

    ModelData(ModelData&& data) noexcept :
            vertex_pyramid(std::move(data.vertex_pyramid)),
            normal_pyramid(std::move(data.normal_pyramid)),
            color_pyramid(std::move(data.color_pyramid))
    { }

    ModelData& operator=(ModelData&& data) noexcept
    {
        vertex_pyramid = std::move(data.vertex_pyramid);
        normal_pyramid = std::move(data.normal_pyramid);
        color_pyramid = std::move(data.color_pyramid);
        return *this;
    }
};