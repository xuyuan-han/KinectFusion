#include "kinectfusion.hpp"

// #define MAXFRAMECNT 1 // Process MAXFRAMECNT frames, comment out this line to process all frames
#define SHOW_IMAGES // Show images
#define OUTPUT_VIDEO // Save video to 'output' folder

int main(int argc, char **argv)
{
    std::cout << "Using GPU CUDA" << std::endl;

    CameraParameters cameraparameters;
    GlobalConfiguration configuration;

    std::string datasetname = std::string("rgbd_dataset_freiburg1_xyz");
    std::string filenameIn = std::string("../../Data/") + datasetname + std::string("/");

    #ifdef OUTPUT_VIDEO
    std::string outputPath = "../output/output_" + datasetname + "/";
    if (!std::filesystem::exists(outputPath)) {
        try {
            if (!std::filesystem::create_directories(outputPath)) {
                std::cerr << "Failed to create output directory: " << outputPath << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double output_video_fps = 30.0;
    cv::Size output_frameSize(640, 480);
    cv::VideoWriter videoWriter_InputRGB(outputPath + "InputRGB.avi", fourcc, output_video_fps, output_frameSize, true);
    cv::VideoWriter videoWriter_InputDepth(outputPath + "InputDepth.avi", fourcc, output_video_fps, output_frameSize, true);
    cv::VideoWriter videoWriter_ModelRGB(outputPath + "ModelRGB.avi", fourcc, output_video_fps, output_frameSize, true);
    cv::VideoWriter videoWriter_ModelNormalMapping(outputPath + "ModelNormalMapping.avi", fourcc, output_video_fps, output_frameSize, true);
    if (!videoWriter_InputRGB.isOpened() || !videoWriter_InputDepth.isOpened() || !videoWriter_ModelRGB.isOpened() || !videoWriter_ModelNormalMapping.isOpened()) {
        std::cerr << "Failed to open video writer" << std::endl;
        return -1;
    }
    #endif

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

        bool success=pipeline.process_frame(sensor.getDepth(), sensor.getColorRGBX());

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
        
        // L.N Shaded rendering
        cv::Mat image_normalMapping = normalMapping(pipeline.get_last_model_normal_frame_in_camera_coordinates(), light, pipeline.get_last_model_vertex_frame());

        std::string fps_text = "FPS: " + std::to_string(int(fps));
        cv::putText(image_last_model_color_frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(image_normalMapping, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0), 3);
        cv::cvtColor(image_normalMapping, image_normalMapping, cv::COLOR_GRAY2BGR);

        cv::Mat depthNormalized;
        sensor.getDepth().convertTo(depthNormalized, CV_8UC1, 255.0 / (5000.f), 0);
        cv::cvtColor(depthNormalized, depthNormalized, cv::COLOR_GRAY2BGR);

    #ifdef SHOW_IMAGES
        cv::imshow("InputRGB", sensor.getColorRGBX());
        cv::moveWindow("InputRGB", 0, 0);

        cv::imshow("InputDepth", depthNormalized);
        cv::moveWindow("InputDepth", sensor.getColorRGBX().cols, 0);
        #ifdef USE_CLASSES
        cv::imshow("InputSegmentation", sensor.getSegmentation());
        cv::moveWindow("InputSegmentation", sensor.getColorRGBX().cols*2, 0);
        #endif
        cv::imshow("ModelRGB", image_last_model_color_frame); // pipeline.get_last_model_color_frame() with FPS
        cv::moveWindow("ModelRGB", 0, sensor.getColorRGBX().rows + 40);

        // L.N Shaded rendering
        cv::imshow("ModelNormalMapping", image_normalMapping);
        cv::moveWindow("ModelNormalMapping", sensor.getColorRGBX().cols, sensor.getColorRGBX().rows + 40);

        // cv::imshow("SurfacePrediction Output: Normal (in camera frame)", pipeline.get_last_model_normal_frame_in_camera_coordinates());
        // cv::moveWindow("SurfacePrediction Output: Normal (in camera frame)", sensor.getColorRGBX().cols * 2, sensor.getColorRGBX().rows + 40);
    #endif
    #ifdef OUTPUT_VIDEO
        videoWriter_InputRGB.write(sensor.getColorRGBX());
        videoWriter_InputDepth.write(depthNormalized);
        videoWriter_ModelRGB.write(image_last_model_color_frame);
        videoWriter_ModelNormalMapping.write(image_normalMapping);
    #endif

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

    #ifdef OUTPUT_VIDEO
    std::cout << "Saving videos..." << std::endl;
    videoWriter_InputDepth.release();
    videoWriter_InputRGB.release();
    videoWriter_ModelRGB.release();
    videoWriter_ModelNormalMapping.release();
    #endif

    std::cout << ">> Point cloud generation begin" << std::endl;
    auto start = std::chrono::high_resolution_clock::now(); // start time measurement
    pipeline.save_tsdf_color_volume_point_cloud();
    auto end_save = std::chrono::high_resolution_clock::now(); // end time measurement
    std::chrono::duration<double, std::milli> elapsed_save = end_save - start; // elapsed time in milliseconds
    std::cout << "-- Save point cloud time: " << elapsed_save.count() << " ms\n";
    std::cout << ">>> Point cloud generation done" << std::endl;
    
    return 0;
}
