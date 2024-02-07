#ifdef HAS_RECORD3D

#include <iostream>
#include <vector>
#include <record3d/Record3DStream.h>
#include <mutex>

#include <opencv2/opencv.hpp>
#include "kinectfusion.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

using namespace std;

#define SHOW_IMAGES // Show images
#define OUTPUT_VIDEO // Save video to 'output' folder

class iPhoneFusion
{
public:
    void Run()
    {
        Record3D::Record3DStream stream { };
        stream.onStreamStopped = [&]
        {
            OnStreamStopped();
        };
        stream.onNewFrame = [&]( const Record3D::BufferRGB &$rgbFrame,
                                 const Record3D::BufferDepth &$depthFrame,
                                 uint32_t $rgbWidth,
                                 uint32_t $rgbHeight,
                                 uint32_t $depthWidth,
                                 uint32_t $depthHeight,
                                 Record3D::DeviceType $deviceType,
                                 Record3D::IntrinsicMatrixCoeffs $K,
                                 Record3D::CameraPose $cameraPose )
        {
            OnNewFrame( $rgbFrame, $depthFrame, $rgbWidth, $rgbHeight, $depthWidth, $depthHeight, $deviceType, $K, $cameraPose );
        };

        // Try connecting to a device.
        const auto &devs = Record3D::Record3DStream::GetConnectedDevices();
        if ( devs.empty() )
        {
            fprintf( stderr,
                     "No iOS devices found. Ensure you have connected your iDevice via USB to this computer.\n" );
            return;
        }
        else
        {
            printf( "Found %lu iOS device(s):\n", devs.size() );
            for ( const auto &dev: devs )
            {
                printf( "\tDevice ID: %u\n\tUDID: %s\n\n", dev.productId, dev.udid.c_str() );
            }
        }

        const auto &selectedDevice = devs[0];
        printf( "Trying to connect to device with ID %u.\n", selectedDevice.productId );

        bool isConnected = stream.ConnectToDevice( devs[0] );
        if ( isConnected )
        {
            printf( "Connected and starting to stream. Enable USB streaming in the Record3D iOS app (https://record3d.app/) in case you don't see RGBD stream.\n" );

            CameraParameters cameraparameters;
            GlobalConfiguration configuration;

            #ifdef OUTPUT_VIDEO
            std::string outputPath = "../output/";
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
                return;
            }
            #endif

            // cameraparameters.focal_x=680.0f;
            // cameraparameters.focal_y=680.0f;
            // cameraparameters.principal_x=360.5f;
            // cameraparameters.principal_y=480.5f;
            // cameraparameters.image_width=720;
            // cameraparameters.image_height=960;

            cameraparameters.focal_x=stream.GetCurrentIntrinsicMatrix()[0];
            cameraparameters.focal_y=stream.GetCurrentIntrinsicMatrix()[1];
            cameraparameters.principal_x=stream.GetCurrentIntrinsicMatrix()[2];
            cameraparameters.principal_y=stream.GetCurrentIntrinsicMatrix()[3];
            cameraparameters.image_width=720;
            cameraparameters.image_height=960;

            Pipeline pipeline {cameraparameters, configuration};

            size_t frameCnt = 0;
            cv::Vec3f light = { 0.0f, 0.0f, 0.0f };
            double last_time = cv::getTickCount();
            int frame_count_FPS = 0;
            double fps = 0.0;

            while ( true )
            // for ( ; frameCnt < 30 * 15; frameCnt++)
            {
                // Wait for the callback thread to receive new frame and unlock this thread
                cv::Mat rgb, depth;
                {
                    std::lock_guard<std::recursive_mutex> lock( mainThreadLock_ );

                    if ( imgRGB_.cols == 0 || imgRGB_.rows == 0 || imgDepth_.cols == 0 || imgDepth_.rows == 0 )
                    {
                        continue;
                    }

                    // print the width and height of the images (w, h) together // RGB: 1440x1920, Depth: 192x256
                    // printf( "RGB: %dx%d, Depth: %dx%d\n", imgRGB_.cols, imgRGB_.rows, imgDepth_.cols, imgDepth_.rows );
                    rgb = imgRGB_.clone();
                    depth = imgDepth_.clone();
                }
                // Postprocess images
                cv::cvtColor( rgb, rgb, cv::COLOR_RGB2BGR );

                // The TrueDepth camera is a selfie camera; we mirror the RGBD frame so it looks plausible.
                if ( currentDeviceType_ == Record3D::R3D_DEVICE_TYPE__FACEID )
                {
                    cv::flip( rgb, rgb, 1 );
                    cv::flip( depth, depth, 1 );
                }

                cameraparameters.focal_x=stream.GetCurrentIntrinsicMatrix()[0];
                cameraparameters.focal_y=stream.GetCurrentIntrinsicMatrix()[1];
                cameraparameters.principal_x=stream.GetCurrentIntrinsicMatrix()[2];
                cameraparameters.principal_y=stream.GetCurrentIntrinsicMatrix()[3];
                cameraparameters.image_width=rgb.cols;
                cameraparameters.image_height=rgb.rows;

                cv::resize(depth, depth, cv::Size(rgb.cols, rgb.rows), 0, 0, cv::INTER_LANCZOS4);

                auto start = std::chrono::high_resolution_clock::now(); // start time measurement

                depth = depth*1000.f; // convert to mm

                bool success=pipeline.process_frame(depth, rgb, cameraparameters);

                auto end = std::chrono::high_resolution_clock::now(); // end time measurement
                std::chrono::duration<double, std::milli> elapsed = end - start; // elapsed time in milliseconds

                #ifdef PRINT_MODULE_COMP_TIME
                std::cout << std::endl;
                #endif

                if (!success){
                    std::cout << ">>> Frame could not be processed" << std::endl;
                    // break;
                }else{
                    std::cout << ">>> Frame processed: " << elapsed.count() << " ms" << std::endl;
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
                depth.convertTo(depthNormalized, CV_8UC1, 255.0 / (5000.f), 0);
                cv::cvtColor(depthNormalized, depthNormalized, cv::COLOR_GRAY2BGR);

            #ifdef SHOW_IMAGES
                cv::imshow("InputRGB", rgb);
                cv::moveWindow("InputRGB", 0, 0);

                cv::imshow("InputDepth", depthNormalized);
                cv::moveWindow("InputDepth", rgb.cols, 0);
                #ifdef USE_CLASSES
                cv::imshow("InputSegmentation", sensor.getSegmentation());
                cv::moveWindow("InputSegmentation", rgb.cols*2, 0);
                #endif
                cv::imshow("ModelRGB", image_last_model_color_frame); // pipeline.get_last_model_color_frame() with FPS
                cv::moveWindow("ModelRGB", 0, rgb.rows + 40);

                // L.N Shaded rendering
                cv::imshow("ModelNormalMapping", image_normalMapping);
                cv::moveWindow("ModelNormalMapping", rgb.cols, rgb.rows + 40);

                // cv::imshow("SurfacePrediction Output: Normal (in camera frame)", pipeline.get_last_model_normal_frame_in_camera_coordinates());
                // cv::moveWindow("SurfacePrediction Output: Normal (in camera frame)", rgb.cols * 2, rgb.rows + 40);
            #endif
            #ifdef OUTPUT_VIDEO
                videoWriter_InputRGB.write(rgb);
                videoWriter_InputDepth.write(depthNormalized);
                videoWriter_ModelRGB.write(image_last_model_color_frame);
                videoWriter_ModelNormalMapping.write(image_normalMapping);
            #endif

                int key = cv::waitKey(1);
                if (key != -1) {
                    break;
                }
            }
            std::cout << "Finished - Total frame processed: " << frameCnt << std::endl;
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
        }
        else
        {
            fprintf( stderr,
                     "Could not connect to iDevice. Make sure you have opened the Record3D iOS app (https://record3d.app/).\n" );
        }
    }

private:
    void OnStreamStopped()
    {
        fprintf( stderr, "Stream stopped!" );
    }

    void OnNewFrame( const Record3D::BufferRGB &$rgbFrame,
                     const Record3D::BufferDepth &$depthFrame,
                     uint32_t $rgbWidth,
                     uint32_t $rgbHeight,
                     uint32_t $depthWidth,
                     uint32_t $depthHeight,
                     Record3D::DeviceType $deviceType,
                     Record3D::IntrinsicMatrixCoeffs $K,
                     Record3D::CameraPose $cameraPose )
    {
        currentDeviceType_ = (Record3D::DeviceType) $deviceType;

        std::lock_guard<std::recursive_mutex> lock( mainThreadLock_ );
        // When we switch between the TrueDepth and the LiDAR camera, the size frame size changes.
        // Recreate the RGB and Depth images with fitting size.
        if ( imgRGB_.rows != $rgbHeight || imgRGB_.cols != $rgbWidth
             || imgDepth_.rows != $depthHeight || imgDepth_.cols != $depthWidth )
        {
            imgRGB_.release();
            imgDepth_.release();

            imgRGB_ = cv::Mat::zeros( $rgbHeight, $rgbWidth, CV_8UC3 );
            imgDepth_ = cv::Mat::zeros( $depthHeight, $depthWidth, CV_32F );
        }

        // The `BufferRGB` and `BufferDepth` may be larger than the actual payload, therefore the true frame size is computed.
        constexpr int numRGBChannels = 3;
        memcpy( imgRGB_.data, $rgbFrame.data(), $rgbWidth * $rgbHeight * numRGBChannels * sizeof( uint8_t ) );
        memcpy( imgDepth_.data, $depthFrame.data(), $depthWidth * $depthHeight * sizeof( float ) );
    }

private:
    std::recursive_mutex mainThreadLock_ { };
    Record3D::DeviceType currentDeviceType_ { };

    cv::Mat imgRGB_ { };
    cv::Mat imgDepth_ { };
};

#pragma clang diagnostic pop

#endif