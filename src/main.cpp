#include "kinectfusion.hpp"

int main(int argc, char **argv)
{
    CameraParameters cameraparameters;
    GlobalConfiguration configuration;
    VirtualSensor sensor;
    Eigen::Matrix3f intrinsics=sensor.getDepthIntrinsics();
    cameraparameters.focal_x=intrinsics(0, 0);
    cameraparameters.focal_y=intrinsics(1, 1);
    cameraparameters.principal_x=intrinsics(0, 2);
    cameraparameters.principal_y=intrinsics(1, 2);
    cameraparameters.image_width=sensor.getDepthImageWidth();
    cameraparameters.image_height=sensor.getDepthImageHeight();

    std::string filenameIn = std::string("../Data/rgbd_dataset_freiburg1_xyz/");

    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    Pipeline pipeline {cameraparameters, configuration};
    while(sensor.processNextFrame()){
        bool success=pipeline.process_frame(sensor.getDepth(), sensor.getColorRGBX());
        if (!success)
            std::cout << "Frame " << sensor.getCurrentFrameCnt() << " could not be processed" << std::endl;
            break;
    }
}

	// FrameData frame=surface_measurement(sensor.getDepth(), cameraparameters,3,1000.f,5,1.0f,1.0f);
	// cv::Mat vertex=frame.vertex_pyramid[0];
	// cv::Mat normal=frame.normal_pyramid[0];
	// cv::Mat rgb=frame.color_pyramid[0];
    // cv::Mat depth=frame.depth_pyramid[0];

	// cv::imwrite("rgb.png", rgb);
    // cv::imwrite("depth.png", depth);
    // cv::imwrite("vertex.png", vertex);

	// cv::viz::Viz3d myWindow("Viz Demo");
    // myWindow.setBackgroundColor(cv::viz::Color::black());

    // // Show coordinate system
    // myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    // // Show point cloud
    // //cv::viz::WCloud pointCloud(normal, cv::viz::Color::green());
    // cv::viz::WCloud pointCloud(vertex, cv::viz::Color::green());
    // myWindow.showWidget("points", pointCloud);
