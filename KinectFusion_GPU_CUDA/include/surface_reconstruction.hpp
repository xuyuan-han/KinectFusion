#pragma once
#ifndef SURFACE_RECON_HPP
#define SURFACE_RECON_HPP

#include "data_types_CPU.hpp"
// #include "data_types_GPU.hpp"

namespace CPU{

namespace Surface_Reconstruction {
	void integrate_multi_threads(cv::Mat depth, cv::Mat colorMap, VolumeData* vol,CameraParameters camera_parameters , float trancutionDistance, Eigen::Matrix4f pos);
	
	void reconstructionProcessVolumeSlice(VolumeData* vol, cv::Mat colorMap, cv::Mat depth, uint* class_map, Eigen::Matrix4f cameraToWorld, Eigen::Matrix3f intrinsics, int width, int height, float trancutionDistance, int zStart, int zEnd);
	
	void surface_reconstruction(cv::Mat depth, cv::Mat colorMap, VolumeData vol,float trancutionDistance, Eigen::Matrix4f pos);

	double getLambda(Eigen::Vector2i pixel, Eigen::Matrix3f intrinsics);
	
	void integrate(cv::Mat depth, cv::Mat colorMap, Volume* vol, CameraParameters camera_parameters, float trancutionDistance, Eigen::Matrix4f pos);
}

#endif // !SURFACE_RECON_HPP

} // namespace CPU