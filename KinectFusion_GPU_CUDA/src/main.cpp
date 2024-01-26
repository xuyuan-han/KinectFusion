#include "kinectfusion.hpp"

// #define MAXFRAMECNT 1 // Process MAXFRAMECNT frames, comment out this line to process all frames

int main(int argc, char **argv)
{

#ifdef USE_CPU_MULTI_THREADING
    std::cout << "Using cpu multi-threading" << std::endl;
#else
    std::cout << "Not using cpu multi-threading" << std::endl;
#endif

    CameraParameters cameraparameters;
    GlobalConfiguration configuration;

    std::string filenameIn = std::string("../../Data/rgbd_dataset_freiburg1_xyz/");

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

    #ifdef MAXFRAMECNT
    unsigned int maxFrameCnt = MAXFRAMECNT;
    #endif

    Pipeline pipeline {cameraparameters, configuration};
    while(sensor.processNextFrame()){

        auto start = std::chrono::high_resolution_clock::now(); // start time measurement

        bool success=pipeline.process_frame(sensor.getDepth(), sensor.getColorRGBX());

        auto end = std::chrono::high_resolution_clock::now(); // end time measurement
        std::chrono::duration<double, std::milli> elapsed = end - start; // elapsed time in milliseconds

        if (!success){
            std::cout << "\n>>> Frame " << sensor.getCurrentFrameCnt() << " could not be processed" << std::endl;
            // break;
        }else{
            std::cout << "\n>>> Frame " << sensor.getCurrentFrameCnt() << " processed: " << elapsed.count() << " ms" << std::endl;
        }
        
        std::cout << "-----------------------------------" << std::endl;

        cv::imshow("InputRGB", sensor.getColorRGBX());
        cv::moveWindow("InputRGB", 0, 0);

        cv::imshow("InputDepth", sensor.getDepth()/5000.f);
        cv::moveWindow("InputDepth", sensor.getColorRGBX().cols, 0);
        
        cv::imshow("SurfacePrediction Output: Color", pipeline.get_last_model_color_frame());
        cv::moveWindow("SurfacePrediction Output: Color", 0, sensor.getColorRGBX().rows + 40);

        cv::imshow("SurfacePrediction Output: Normal (in camera frame)", pipeline.get_last_model_normal_frame_in_camera_coordinates());
        cv::moveWindow("SurfacePrediction Output: Normal (in camera frame)", sensor.getColorRGBX().cols, sensor.getColorRGBX().rows + 40);

        cv::waitKey(1);

        #ifdef MAXFRAMECNT
        if (sensor.getCurrentFrameCnt() == (maxFrameCnt-1)) {
            break;
        }
        #endif
    }
    std::cout << "Finished - Total frame processed: " << sensor.getCurrentFrameCnt() << std::endl;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << ">> Point cloud generation begin" << std::endl;
    auto start = std::chrono::high_resolution_clock::now(); // start time measurement
    pipeline.save_tsdf_color_volume_point_cloud();
    auto end_save = std::chrono::high_resolution_clock::now(); // end time measurement
    std::chrono::duration<double, std::milli> elapsed_save = end_save - start; // elapsed time in milliseconds
    std::cout << "-- Save point cloud time: " << elapsed_save.count() << " ms\n";
    std::cout << ">>> Point cloud generation done" << std::endl;

    cv::waitKey(1);
}
