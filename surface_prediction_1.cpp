// #include "include/common.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

using Vec3ida = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            // 三线型插值
            float interpolate_trilinearly(
                const Vec3fda& point,
                const std::vector<short2>& volume,  // TSDF Volume，更改为 std::vector 或适合的数据结构
                const int3& volume_size,
                const float voxel_scale)
                {
                        // 这个点在 Volume 下的坐标, 转换成为整数下标标的表示
                    Vec3ida point_in_grid = point.cast<int>();

                    // 恢复成体素中心点的坐标
                    const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
                    const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
                    const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

                        // 分成这两种情况是为了方便计算不同组织形式下的插值        
                    point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
                    point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
                    point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

                    // +0.5f 的原因是, point_in_grid 处体素存储的TSDF值是体素的中心点的TSDF值
                    // 三线型插值, ref: https://en.wikipedia.org/wiki/Trilinear_interpolation
                    // 计算精确的(浮点型)的点坐标和整型化之后的点坐标的差
                    const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
                    const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
                    const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

                    return 
                        static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX 
                            // volume[ x ][ y ][ z ], C000
                            * (1 - a) * (1 - b) * (1 - c) +
                        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX 
                            // volume[ x ][ y ][z+1], C001
                            * (1 - a) * (1 - b) * c +
                        static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX 
                            // volume[ x ][y+1][ z ], C010
                            * (1 - a) * b * (1 - c) +
                        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX 
                            // volume[ x ][y+1][z+1], C011
                            * (1 - a) * b * c +
                        static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                            // volume[x+1][ y ][ z ], C100
                            * a * (1 - b) * (1 - c) +
                        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                            // volume[x+1][ y ][z+1], C101
                            * a * (1 - b) * c +
                        static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                            // volume[x+1][y+1][ z ], C110
                            * a * b * (1 - c) +
                        static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX 
                            // volume[x+1][y+1][z+1], C111
                            * a * b * c;
                }

            float get_min_time(
                const float3& volume_max,
                const Vec3fda& origin,
                const Vec3fda& direction)
            {
                // 分别计算三个轴上的次数, 并且返回其中最大; 当前进了这个最大的次数之后, 三个轴上射线的分量就都已经射入volume了
                float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
                float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
                float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();
                
                return fmax(fmax(txmin, tymin), tzmin);
            }

            float get_max_time(
                const float3& volume_max,
                const Vec3fda& origin,
                const Vec3fda& direction
            )
            {
                // 分别计算三个轴上的次数, 并且返回其中最小; 当前进了这个最小的次数之后, 三个轴上射线的分量就都已经射入volume了
                float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
                float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
                float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();
                
                return fmin(fmin(txmax, tymax), tzmax);
            }

            void process_pixel(
                int x, int y,
                const std::vector<short2>& tsdf_volume,
                const std::vector<uchar3>& color_volume,
                std::vector<float3>& model_vertex,
                std::vector<float3>& model_normal,
                std::vector<uchar3>& model_color,
                const int3& volume_size,
                const float voxel_scale,
                const CameraParameters& cam_parameters,
                const float truncation_distance,
                const Eigen::Matrix<float, 3, 3, Eigen::DontAlign>& rotation,
                const Vec3fda& translation)
            {

            }

            void surface_prediction(
                // [其他参数保持不变]
            )
            {
                // [去除 CUDA 相关代码]

                for (int y = 0; y < image_height; ++y) {
                    for (int x = 0; x < image_width; ++x) {
                        process_pixel(x, y, /* 其他参数 */);
                    }
                }
            }

        }  // namespace cuda
    }  // namespace internal
}  // namespace kinectfusion   