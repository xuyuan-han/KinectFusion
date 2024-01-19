#include "pose_estimation.hpp"

bool pose_estimation(Eigen::Matrix4f& pose,
                    const FrameData& frame_data,
                    const ModelData& model_data,
                    const CameraParameters& cam_params,
                    const int pyramid_height,
                    const float distance_threshold, 
                    const float angle_threshold,
                    const std::vector<int>& iterations){

    // Get initial rotation and translation
    Eigen::Matrix3f current_global_rotation = pose.block(0, 0, 3, 3);
    Eigen::Vector3f current_global_translation = pose.block(0, 3, 3, 1);

    Eigen::Matrix3f previous_global_rotation_inverse(current_global_rotation.inverse());
    Eigen::Vector3f previous_global_translation = pose.block(0, 3, 3, 1);

    // ICP loop, from coarse to sparse
    for (int level = pyramid_height - 1; level >= 0; --level) {
        for (int iteration = 0; iteration < iterations[level]; ++iteration) {
            // std::cout << "-------------" << std::endl;
            // std::cout << "ICP level: " << level << " iteration: " << iteration << std::endl;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A {};
            Eigen::Matrix<double, 6, 1> b {};

            // std::cout << "ICP before estimate step" << std::endl;

            // Estimate one step on the CPU
            // estimate_step(current_global_rotation, current_global_translation,
            //                     frame_data.vertex_pyramid[level], frame_data.normal_pyramid[level],
            //                     previous_global_rotation_inverse, previous_global_translation,
            //                     cam_params.level(level),
            //                     model_data.vertex_pyramid[level], model_data.normal_pyramid[level],
            //                     distance_threshold, sinf(angle_threshold * 3.14159254f / 180.f),
            //                     A, b);
            estimate_step_multi_threads(current_global_rotation, current_global_translation,
                                frame_data.vertex_pyramid[level], frame_data.normal_pyramid[level],
                                previous_global_rotation_inverse, previous_global_translation,
                                cam_params.level(level),
                                model_data.vertex_pyramid[level], model_data.normal_pyramid[level],
                                distance_threshold, sinf(angle_threshold * 3.14159254f / 180.f),
                                A, b);

            // std::cout << "ICP after estimate step" << std::endl;

            // std::cout << "A: \n" << std::fixed << std::setprecision(2) << A << std::endl;
            // std::cout << "b: \n" << std::fixed << std::setprecision(2) << b << std::endl;
            // std::cout << std::defaultfloat;

            // Directly solve the linear system A * x = b to get alpha, beta and gamma
            Eigen::Matrix<double, 6, 1> result_double = A.fullPivLu().solve(b);

            // Verify if the solution satisfies the original equation system
            Eigen::Matrix<double, 6, 1> residual = A * result_double - b;

            // Cast result to float for further use
            Eigen::Matrix<float, 6, 1> result = result_double.cast<float>();

            // Set a threshold for checking the validity of the solution
            double tolerance = 1e-5;
            if (residual.norm() > tolerance) {
                std::cout << "The solution does not satisfy the original equation system, indicating possible numerical issues or near-singularity of matrix A." << std::endl;
                return false;
            }

            float alpha = result(0);
            float beta = result(1);
            float gamma = result(2);

            // std::cout << "ICP after LU" << std::endl;

            // Update rotation
            auto camera_rotation_incremental(
                    Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()) *
                    Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()) *
                    Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()));
            auto camera_translation_incremental = result.tail<3>();

            // std::cout << "ICP after incremental" << std::endl;

            // Update translation
            current_global_translation =
                    camera_rotation_incremental * current_global_translation + camera_translation_incremental;
            current_global_rotation = camera_rotation_incremental * current_global_rotation;

            Eigen::Matrix4f current_global_pose = Eigen::Matrix4f::Identity();
            current_global_pose.block(0, 0, 3, 3) = current_global_rotation;
            current_global_pose.block(0, 3, 3, 1) = current_global_translation;
            // std::cout << "Pose: \n" << current_global_pose << std::endl;
            // std::cout << "Pose: \n" << std::fixed << std::setprecision(2) << current_global_pose << std::endl;
            // std::cout << std::defaultfloat;
        }
    }

    // std::cout << "ICP after pyramid loop" << std::endl;

    // Return the new pose
    pose.block(0, 0, 3, 3) = current_global_rotation;
    pose.block(0, 3, 3, 1) = current_global_translation;
    return true;
}

void estimate_step(const Eigen::Matrix3f& rotation_current, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current,
                const cv::Mat& vertex_map_current, 
                const cv::Mat& normal_map_current,
                const Eigen::Matrix3f& rotation_previous_inv, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous,
                const CameraParameters& cam_params,
                const cv::Mat& vertex_map_previous, 
                const cv::Mat& normal_map_previous,
                float distance_threshold, 
                float angle_threshold,
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, 
                Eigen::Matrix<double, 6, 1>& b){

    const int cols = vertex_map_current.cols;
    const int rows = vertex_map_current.rows;
    A = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
    b = Eigen::Matrix<double, 6, 1>::Zero();

    for (int x = 0; x < cols; ++x) {
        for (int y = 0; y < rows; ++y) {
    // for (int x = 0; x < 50; ++x) {
    //     for (int y = 0; y < 50; ++y) {
            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> n, d, s;
            bool correspondence_found = false;

            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> normal_current;
            normal_current.x() = normal_map_current.at<cv::Vec3f>(y, x)[0];

            if (!isnan(normal_current.x())) {
                // Get current point
                Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_current;
                vertex_current.x() = vertex_map_current.at<cv::Vec3f>(y, x)[0];
                vertex_current.y() = vertex_map_current.at<cv::Vec3f>(y, x)[1];
                vertex_current.z() = vertex_map_current.at<cv::Vec3f>(y, x)[2];

                Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_current_global = rotation_current * vertex_current + translation_current;

                Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_current_camera = rotation_previous_inv * (vertex_current_global - translation_previous);

                Eigen::Vector2i point;
                point.x() = static_cast<int>(roundf(vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() + cam_params.principal_x));
                point.y() = static_cast<int>(roundf(vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() + cam_params.principal_y));

                if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows && vertex_current_camera.z() >= 0){
                    Eigen::Matrix<float, 3, 1, Eigen::DontAlign> normal_previous_global;
                    normal_previous_global.x() = normal_map_previous.at<cv::Vec3f>(point.y(), point.x())[0];
                    if (!isnan(normal_previous_global.x())){
                        Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_previous_global;
                        vertex_previous_global.x() = vertex_map_previous.at<cv::Vec3f>(point.y(), point.x())[0];
                        vertex_previous_global.y() = vertex_map_previous.at<cv::Vec3f>(point.y(), point.x())[1];
                        vertex_previous_global.z() = vertex_map_previous.at<cv::Vec3f>(point.y(), point.x())[2];
                        float distance = (vertex_previous_global - vertex_current_global).norm();

                        // std::cout << "distance: " << distance << std::endl;

                        if (distance <= distance_threshold){
                            normal_current.y() = normal_map_current.at<cv::Vec3f>(y, x)[1];
                            normal_current.z() = normal_map_current.at<cv::Vec3f>(y, x)[2];
                            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> normal_current_global = rotation_current * normal_current;

                            normal_previous_global.y() = normal_map_previous.at<cv::Vec3f>(point.y(), point.x())[1];
                            normal_previous_global.z() = normal_map_previous.at<cv::Vec3f>(point.y(), point.x())[2];

                            float angle = acosf(normal_previous_global.dot(normal_current_global));
                            
                            // std::cout << "angle: " << angle << std::endl;
                            
                            if (angle <= angle_threshold){
                                n = normal_previous_global;
                                d = vertex_previous_global;
                                s = vertex_current_global;

                                correspondence_found = true;
                                
                                // std::cout << "ICP estimate_step correspondence found" << std::endl;
                            }
                        }
                    }
                }
            }
            if (correspondence_found){
                Eigen::Vector<float, 6> vec6f;
                vec6f.head<3>() = s.cross(n);
                vec6f.tail<3>() = n;

                // std::cout << "vec6f: " << vec6f << std::endl;
                // std::cout << "n: " << n << std::endl;
                // std::cout << "d: " << d << std::endl;
                // std::cout << "s: " << s << std::endl;

                //      | 0x0 0x1 0x2 0x3 0x4 0x5 |
                //      | 1x0 1x1 1x2 1x3 1x4 1x5 |
                //      | 2x0 2x1 2x2 2x3 2x4 2x5 |
                // A =  | 3x0 3x1 3x2 3x3 3x4 3x5 |
                //      | 4x0 4x1 4x2 4x3 4x4 4x5 |
                //      | 5x0 5x1 5x2 5x3 5x4 5x5 |

                //      | 0x6 |
                //      | 1x6 |
                //      | 2x6 |
                // b =  | 3x6 |
                //      | 4x6 |
                //      | 5x6 |

                A += (vec6f.cast<double>()) * (vec6f.transpose().cast<double>());
                b += (vec6f.cast<double>()) * (n.dot(d - s));

                // std::cout << "A*:\n" << (vec6f.cast<double>()) * (vec6f.transpose().cast<double>()) << std::endl;
                // std::cout << "b*:\n" << (vec6f.cast<double>()) * (n.dot(d - s)) << std::endl;
            }
            else{
                // std::cout << "ICP estimate_step correspondence not found" << std::endl;
            }
        }
    }
}

void estimate_step_multi_threads(const Eigen::Matrix3f& rotation_current, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current,
                const cv::Mat& vertex_map_current, 
                const cv::Mat& normal_map_current,
                const Eigen::Matrix3f& rotation_previous_inv, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous,
                const CameraParameters& cam_params,
                const cv::Mat& vertex_map_previous, 
                const cv::Mat& normal_map_previous,
                float distance_threshold, 
                float angle_threshold,
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, 
                Eigen::Matrix<double, 6, 1>& b){

    const int cols = vertex_map_current.cols;
    const int rows = vertex_map_current.rows;
    A = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
    b = Eigen::Matrix<double, 6, 1>::Zero();

    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    std::vector<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> A_vec{numThreads, Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()};
    std::vector<Eigen::Matrix<double, 6, 1>> b_vec{numThreads, Eigen::Matrix<double, 6, 1>::Zero()};

    int rowsPerThread = rows / numThreads;
    for (int i = 0; i < numThreads; ++i){
        int rowStart = i * rowsPerThread;
        int rowEnd = (i == numThreads - 1) ? rows : (i + 1) * rowsPerThread;
        threads[i] = std::thread(estimate_step_pixel_slice, 
                                cols, rows, rowStart, rowEnd,
                                std::ref(rotation_current), std::ref(translation_current),
                                std::ref(vertex_map_current), std::ref(normal_map_current),
                                std::ref(rotation_previous_inv), std::ref(translation_previous),
                                std::ref(cam_params),
                                std::ref(vertex_map_previous), std::ref(normal_map_previous),
                                distance_threshold, angle_threshold,
                                std::ref(A_vec[i]), std::ref(b_vec[i])
                                );
    }

    for (auto& thread : threads){
        thread.join();
    }

    for (int i = 0; i < numThreads; ++i){
        A += A_vec[i];
        b += b_vec[i];
    }
}

void estimate_step_pixel_slice(int cols, int rows, int rowStart, int rowEnd,
                const Eigen::Matrix3f& rotation_current, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_current,
                const cv::Mat& vertex_map_current, 
                const cv::Mat& normal_map_current,
                const Eigen::Matrix3f& rotation_previous_inv, 
                const Eigen::Matrix<float, 3, 1, Eigen::DontAlign>& translation_previous,
                const CameraParameters& cam_params,
                const cv::Mat& vertex_map_previous, 
                const cv::Mat& normal_map_previous,
                float distance_threshold, 
                float angle_threshold,
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, 
                Eigen::Matrix<double, 6, 1>& b){
    for (int x = 0; x < cols; ++x) {
        for (int y = rowStart; y < rowEnd; ++y) {
            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> n, d, s;
            bool correspondence_found = false;

            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> normal_current;
            normal_current.x() = normal_map_current.at<cv::Vec3f>(y, x)[0];

            if (!isnan(normal_current.x())) {
                // Get current point
                Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_current;
                vertex_current.x() = vertex_map_current.at<cv::Vec3f>(y, x)[0];
                vertex_current.y() = vertex_map_current.at<cv::Vec3f>(y, x)[1];
                vertex_current.z() = vertex_map_current.at<cv::Vec3f>(y, x)[2];

                Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_current_global = rotation_current * vertex_current + translation_current;

                Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_current_camera = rotation_previous_inv * (vertex_current_global - translation_previous);

                Eigen::Vector2i point;
                point.x() = static_cast<int>(roundf(vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() + cam_params.principal_x));
                point.y() = static_cast<int>(roundf(vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() + cam_params.principal_y));

                if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows && vertex_current_camera.z() >= 0){
                    Eigen::Matrix<float, 3, 1, Eigen::DontAlign> normal_previous_global;
                    normal_previous_global.x() = normal_map_previous.at<cv::Vec3f>(point.y(), point.x())[0];
                    if (!isnan(normal_previous_global.x())){
                        Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vertex_previous_global;
                        vertex_previous_global.x() = vertex_map_previous.at<cv::Vec3f>(point.y(), point.x())[0];
                        vertex_previous_global.y() = vertex_map_previous.at<cv::Vec3f>(point.y(), point.x())[1];
                        vertex_previous_global.z() = vertex_map_previous.at<cv::Vec3f>(point.y(), point.x())[2];
                        float distance = (vertex_previous_global - vertex_current_global).norm();

                        // std::cout << "distance: " << distance << std::endl;

                        if (distance <= distance_threshold){
                            normal_current.y() = normal_map_current.at<cv::Vec3f>(y, x)[1];
                            normal_current.z() = normal_map_current.at<cv::Vec3f>(y, x)[2];
                            Eigen::Matrix<float, 3, 1, Eigen::DontAlign> normal_current_global = rotation_current * normal_current;

                            normal_previous_global.y() = normal_map_previous.at<cv::Vec3f>(point.y(), point.x())[1];
                            normal_previous_global.z() = normal_map_previous.at<cv::Vec3f>(point.y(), point.x())[2];

                            float angle = acosf(normal_previous_global.dot(normal_current_global));
                            
                            // std::cout << "angle: " << angle << std::endl;
                            
                            if (angle <= angle_threshold){
                                n = normal_previous_global;
                                d = vertex_previous_global;
                                s = vertex_current_global;

                                correspondence_found = true;
                                
                                // std::cout << "ICP estimate_step correspondence found" << std::endl;
                            }
                        }
                    }
                }
            }
            if (correspondence_found){
                Eigen::Vector<float, 6> vec6f;
                vec6f.head<3>() = s.cross(n);
                vec6f.tail<3>() = n;

                // std::cout << "vec6f: " << vec6f << std::endl;
                // std::cout << "n: " << n << std::endl;
                // std::cout << "d: " << d << std::endl;
                // std::cout << "s: " << s << std::endl;

                //      | 0x0 0x1 0x2 0x3 0x4 0x5 |
                //      | 1x0 1x1 1x2 1x3 1x4 1x5 |
                //      | 2x0 2x1 2x2 2x3 2x4 2x5 |
                // A =  | 3x0 3x1 3x2 3x3 3x4 3x5 |
                //      | 4x0 4x1 4x2 4x3 4x4 4x5 |
                //      | 5x0 5x1 5x2 5x3 5x4 5x5 |

                //      | 0x6 |
                //      | 1x6 |
                //      | 2x6 |
                // b =  | 3x6 |
                //      | 4x6 |
                //      | 5x6 |

                A += (vec6f.cast<double>()) * (vec6f.transpose().cast<double>());
                b += (vec6f.cast<double>()) * (n.dot(d - s));

                // std::cout << "A*:\n" << (vec6f.cast<double>()) * (vec6f.transpose().cast<double>()) << std::endl;
                // std::cout << "b*:\n" << (vec6f.cast<double>()) * (n.dot(d - s)) << std::endl;
            }
            else{
                // std::cout << "ICP estimate_step correspondence not found" << std::endl;
            }
        }
    }
}