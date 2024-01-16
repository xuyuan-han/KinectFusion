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

        // for (int i = 0; i < sensor.getDepth().rows; ++i) {
        //     for (int j = 0; j < 5; ++j) {
        //         sensor.processNextFrame();
        //     }
        // }
        // std::cout << sensor.getDepth() << std::endl;

        bool success=pipeline.process_frame(sensor.getDepth(), sensor.getColorRGBX());

        if (!success){
            std::cout << "Frame " << sensor.getCurrentFrameCnt() << " could not be processed" << std::endl;
            break;
        }else{
            std::cout << "Frame " << sensor.getCurrentFrameCnt() << " processed" << std::endl;
        }

        cv::imshow("InputRGB", sensor.getColorRGBX());
        cv::moveWindow("InputRGB", 0, 0);

        cv::imshow("InputDepth", sensor.getDepth());
        cv::moveWindow("InputDepth", sensor.getColorRGBX().cols, 0);
        
        cv::imshow("Pipeline Output: Model", pipeline.get_last_model_color_frame());
        cv::moveWindow("Pipeline Output: Model", 0, sensor.getColorRGBX().rows + 40);
        
        cv::imshow("Pipeline Output: Vertex", pipeline.get_last_model_vertex_frame());
        cv::moveWindow("Pipeline Output: Vertex", sensor.getColorRGBX().cols, sensor.getColorRGBX().rows + 40);

        cv::imshow("Pipeline Output: Normal", pipeline.get_last_model_normal_frame());
        cv::moveWindow("Pipeline Output: Normal", 2 * sensor.getColorRGBX().cols, sensor.getColorRGBX().rows + 40);

        std::string filenameOut = std::string("../output/");
        cv::imwrite(filenameOut + "InputRGB_" + std::to_string(sensor.getCurrentFrameCnt()) + ".png", sensor.getColorRGBX());
        cv::imwrite(filenameOut + "InputDepth_" + std::to_string(sensor.getCurrentFrameCnt()) + ".png", sensor.getDepth());
        cv::imwrite(filenameOut + "PipelineOutputModel_" + std::to_string(sensor.getCurrentFrameCnt()) + ".png", pipeline.get_last_model_color_frame());
        cv::imwrite(filenameOut + "PipelineOutputVertex_" + std::to_string(sensor.getCurrentFrameCnt()) + ".png", pipeline.get_last_model_vertex_frame());
        cv::imwrite(filenameOut + "PipelineOutputNormal_" + std::to_string(sensor.getCurrentFrameCnt()) + ".png", pipeline.get_last_model_normal_frame());

        cv::waitKey(0);
    }
}
