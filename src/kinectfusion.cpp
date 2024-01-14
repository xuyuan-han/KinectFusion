#include "kinectfusion.hpp"

namespace kinectfusion {
    Pipeline::Pipeline(const CameraParameters _camera_parameters,
                       const GlobalConfiguration _configuration) :
            camera_parameters(_camera_parameters),
            configuration(_configuration),
            volume(_configuration.volume_size, _configuration.voxel_scale), //TODO: check the init parameters
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

        Surface_Reconstruction::integrate(
            frame_data.depth_pyramid[0],
            frame_data.color_pyramid[0],
            &volume,
            camera_parameters,
            configuration.truncation_distance,
            current_pose);

        volumedata.tsdf_volume = volume.getVolume();
        volumedata.color_volume = volume.getColorVolume();
        for (int level = 0; level < configuration.num_levels; ++level)
            surface_prediction(
                volumedata,
                model_data.vertex_pyramid[level],
                model_data.normal_pyramid[level],
                model_data.color_pyramid[level],
                camera_parameters.level(level),
                configuration.truncation_distance,
                current_pose);
        ++frame_id;
        return true;
    }
    std::vector<Eigen::Matrix4f> Pipeline::get_poses() const
    {
        for (auto pose : poses)
            pose.block(0, 0, 3, 3) = pose.block(0, 0, 3, 3).inverse();
        return poses;
    }
}