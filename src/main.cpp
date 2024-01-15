#include "kinectfusion.hpp"

int main(int argc, char **argv)
{
    CameraParameters cameraparameters;
    GlobalConfiguration configuration;

    std::string filenameIn = std::string("../Data/rgbd_dataset_freiburg1_xyz/");

    VirtualSensor sensor;
    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    Eigen::Matrix3f intrinsics=sensor.getDepthIntrinsics();
    cameraparameters.focal_x=intrinsics(0, 0);
    cameraparameters.focal_y=intrinsics(1, 1);
    cameraparameters.principal_x=intrinsics(0, 2);
    cameraparameters.principal_y=intrinsics(1, 2);
    cameraparameters.image_width=sensor.getDepthImageWidth();
    cameraparameters.image_height=sensor.getDepthImageHeight();

    Pipeline pipeline {cameraparameters, configuration};
    while(sensor.processNextFrame()){
        bool success=pipeline.process_frame(sensor.getDepth(), sensor.getColorRGBX());
        if (!success)
            std::cout << "Frame " << sensor.getCurrentFrameCnt() << " could not be processed" << std::endl;
            break;
        cv::imshow("Pipeline Output", pipeline.get_last_model_frame());
        cv::waitKey(5000); // Wait 5 seconds
    }
}
