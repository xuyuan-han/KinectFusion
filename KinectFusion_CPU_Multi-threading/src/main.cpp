#include "kinectfusion.hpp"

// #define MAXFRAMECNT 300 // Process MAXFRAMECNT frames, comment out this line to process all frames

int main(int argc, char **argv)
{

#ifdef USE_CPU_MULTI_THREADING
    std::cout << "Using cpu multi-threading" << std::endl;
#else
    std::cout << "Not using cpu multi-threading" << std::endl;
#endif

    CameraParameters cameraparameters;
    GlobalConfiguration configuration;

    std::string datasetname = std::string("rgbd_dataset_freiburg1_xyz");
    std::string filenameIn = std::string("../../Data/") + datasetname + std::string("/");

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
    cv::Vec3f light = { 0.0f, 0.0f, 0.0f };
    double last_time = cv::getTickCount();
    int frame_count_FPS = 0;
    double fps = 0.0;

    Pipeline pipeline {cameraparameters, configuration, datasetname};
    while(sensor.processNextFrame()){

        auto start = std::chrono::high_resolution_clock::now(); // start time measurement

        bool success=pipeline.process_frame(sensor.getDepth(), sensor.getColorRGBX(), sensor.getSegmentation());

        auto end = std::chrono::high_resolution_clock::now(); // end time measurement
        std::chrono::duration<double, std::milli> elapsed = end - start; // elapsed time in milliseconds
        
        #ifdef PRINT_MODULE_COMP_TIME
        std::cout << std::endl;
        #endif

        if (!success){
            std::cout << ">>> Frame " << sensor.getCurrentFrameCnt() << " could not be processed" << std::endl;
            // break;
        }else{
            std::cout << ">>> Frame " << sensor.getCurrentFrameCnt() << " processed: " << elapsed.count() << " ms" << std::endl;
        }
        
        #ifdef PRINT_MODULE_COMP_TIME
        std::cout << "-----------------------------------" << std::endl;
        #endif

        frame_count_FPS++;
        double current_time = cv::getTickCount();
        double time_diff = (current_time - last_time) / cv::getTickFrequency();

        // update FPS every second
        if (time_diff >= 1.0) {
            fps = frame_count_FPS / time_diff;
            frame_count_FPS = 0;
            last_time = current_time;
        }
    
        cv::Mat image_last_model_color_frame = pipeline.get_last_model_color_frame();
        cv::Mat image_normalMapping = normalMapping(pipeline.get_last_model_normal_frame_in_camera_coordinates(), light, pipeline.get_last_model_vertex_frame());
        std::string fps_text = "FPS: " + std::to_string(int(fps));
        cv::putText(image_last_model_color_frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(image_normalMapping, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0), 3);

        cv::imshow("InputRGB", sensor.getColorRGBX());
        cv::moveWindow("InputRGB", 0, 0);


        cv::imshow("InputDepth", sensor.getDepth()/5000.f);
        cv::moveWindow("InputDepth", sensor.getColorRGBX().cols, 0);
        #ifdef USE_CLASSES
        cv::imshow("InputSegmentation", sensor.getSegmentation());
        cv::moveWindow("InputSegmentation", sensor.getColorRGBX().cols*2, 0);
        #endif
        cv::imshow("SurfacePrediction Output: Color", image_last_model_color_frame); // pipeline.get_last_model_color_frame() with FPS
        cv::moveWindow("SurfacePrediction Output: Color", 0, sensor.getColorRGBX().rows + 40);

        cv::imshow("normal Mapping", image_normalMapping);
        cv::moveWindow("normal Mapping", sensor.getColorRGBX().cols, sensor.getColorRGBX().rows + 40);

        // cv::imshow("SurfacePrediction Output: Normal (in camera frame)", pipeline.get_last_model_normal_frame_in_camera_coordinates());
        // cv::moveWindow("SurfacePrediction Output: Normal (in camera frame)", sensor.getColorRGBX().cols * 2, sensor.getColorRGBX().rows + 40);

        int key = cv::waitKey(1);
        if (key != -1) {
            break;
        }

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
}
