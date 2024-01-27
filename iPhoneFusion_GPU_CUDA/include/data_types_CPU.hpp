#pragma once
#include "data_types_GPU.hpp"

namespace CPU{
struct FrameData {
	std::vector<cv::Mat> depth_pyramid;
	std::vector<cv::Mat> smoothed_depth_pyramid;
	std::vector<cv::Mat> color_pyramid;

	std::vector<cv::Mat> vertex_pyramid;
	std::vector<cv::Mat> normal_pyramid;

	explicit FrameData(const size_t pyramid_height) :
		depth_pyramid(pyramid_height), smoothed_depth_pyramid(pyramid_height),
		color_pyramid(pyramid_height), vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height)
	{ }

	// No copying
	FrameData(const FrameData&) = delete;
	FrameData& operator=(const FrameData& other) = delete;

	FrameData(FrameData&& data) noexcept :
		depth_pyramid(std::move(data.depth_pyramid)),
		smoothed_depth_pyramid(std::move(data.smoothed_depth_pyramid)),
		color_pyramid(std::move(data.color_pyramid)),
		vertex_pyramid(std::move(data.vertex_pyramid)),
		normal_pyramid(std::move(data.normal_pyramid))
	{ }

	FrameData& operator=(FrameData&& data) noexcept
	{
		depth_pyramid = std::move(data.depth_pyramid);
		smoothed_depth_pyramid = std::move(data.smoothed_depth_pyramid);
		color_pyramid = std::move(data.color_pyramid);
		vertex_pyramid = std::move(data.vertex_pyramid);
		normal_pyramid = std::move(data.normal_pyramid);
		return *this;
	}
};

struct ModelData {
	std::vector<cv::Mat> vertex_pyramid;
	std::vector<cv::Mat> normal_pyramid;
	std::vector<cv::Mat> color_pyramid;

	ModelData(const size_t pyramid_height, const CameraParameters camera_parameters) :
		vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height),
		color_pyramid(pyramid_height)
	{
		for (size_t level = 0; level < pyramid_height; ++level) {
			vertex_pyramid[level] =
				cv::Mat(camera_parameters.level(level).image_height,
					camera_parameters.level(level).image_width,
					CV_32FC3);
			normal_pyramid[level] =
				cv::Mat(camera_parameters.level(level).image_height,
					camera_parameters.level(level).image_width,
					CV_32FC3);
			color_pyramid[level] =
				cv::Mat(camera_parameters.level(level).image_height,
					camera_parameters.level(level).image_width,
					CV_8UC3);
			vertex_pyramid[level].setTo(0);
			normal_pyramid[level].setTo(0);
		}
	}

	// No copying
	ModelData(const ModelData&) = delete;
	ModelData& operator=(const ModelData& data) = delete;

	ModelData(ModelData&& data) noexcept :
		vertex_pyramid(std::move(data.vertex_pyramid)),
		normal_pyramid(std::move(data.normal_pyramid)),
		color_pyramid(std::move(data.color_pyramid))
	{ }

	ModelData& operator=(ModelData&& data) noexcept
	{
		vertex_pyramid = std::move(data.vertex_pyramid);
		normal_pyramid = std::move(data.normal_pyramid);
		color_pyramid = std::move(data.color_pyramid);
		return *this;
	}
};

struct Voxel {
	double sdf; // signed distance function
	double weight; // weight for averaging
	Vector4uc color; // color
	uint class_id;

	Voxel() : sdf(0), weight(0), color(0, 0, 0, 0), class_id(0) { }

};

class Volume
{

private:
	//! Lower left and Upper right corner.
	Eigen::Vector3d min, max;

	//! max-min
	Eigen::Vector3d diag;

	double ddx, ddy, ddz;
	double dddx, dddy, dddz;

	//! Number of cells in x, y and z-direction.
	uint dx, dy, dz;

	Voxel* vol;

	double maxValue, minValue;

	uint m_dim;

	//! x,y,z access to vol*
	inline Voxel vol_access(int x, int y, int z) const
	{
		return vol[getPosFromTuple(x, y, z)];
	}
public:
	//! Initializes the volume.
	Volume(Eigen::Vector3d min_, Eigen::Vector3d max_, uint dx_, uint dy_, uint dz_, uint dim)
	{
		min = min_;
		max = max_;
		diag = max - min;
		dx = dx_;
		dy = dy_;
		dz = dz_;
		m_dim = dim;
		vol = NULL;

		vol = new Voxel[dx * dy * dz];

		compute_ddx_dddx();
	}
	Volume(const Eigen::Vector3i _volume_size, const float _voxel_scale)
	{
		// min = Eigen::Vector3d(-_volume_size[0] / 2, -_volume_size[1] / 2, -_volume_size[2] / 2);
		min = Eigen::Vector3d(0,0,0);
		min = min * _voxel_scale;
		// max = Eigen::Vector3d(_volume_size[0] / 2, _volume_size[1] / 2, _volume_size[2] / 2);
		max = Eigen::Vector3d(_volume_size[0], _volume_size[1], _volume_size[2]);
		max = max * _voxel_scale;
		diag = max - min;
		dx = _volume_size[0];
		dy = _volume_size[1];
		dz = _volume_size[2];
		m_dim = 3;
		vol = NULL;

		vol = new Voxel[dx * dy * dz];

		compute_ddx_dddx();
	}

	~Volume()
	{
		delete[] vol;
	};


	//! Computes spacing in x,y,z-directions.
	void compute_ddx_dddx()
	{
		ddx = 1.0f / (dx - 1); //This is how much the x value in voxel space changes per voxel assuming the volume is normalized to 1
		ddy = 1.0f / (dy - 1);
		ddz = 1.0f / (dz - 1);

		dddx = (max[0] - min[0]) / (dx - 1); //This is how much the x value in real space changes per voxel
		dddy = (max[1] - min[1]) / (dy - 1);
		dddz = (max[2] - min[2]) / (dz - 1);

		if (dz == 1)
		{
			ddz = 0;
			dddz = 0;
		}

		diag = max - min;
	}

	//! Zeros out the memory
	void zeroOutMemory()
	{
		for (uint i1 = 0; i1 < dx * dy * dz; i1++)
			vol[i1].sdf = double(0.0);
	}

	//! Returns the Data.
	Voxel* getData()
	{
		return vol;
	};

	//! Sets all entries in the volume to '0'
	void clean()
	{
		for (uint i1 = 0; i1 < dx * dy * dz; i1++) vol[i1].sdf = double(0.0);
	}

	//! Sets minimum extension
	void SetMin(Eigen::Vector3d min_)
	{
		min = min_;
		diag = max - min;
	}

	//! Sets maximum extension
	void SetMax(Eigen::Vector3d max_)
	{
		max = max_;
		diag = max - min;
	}
	//! finds the minimum and maximum values in the volume
	inline void computeMinMaxValues(double& minVal, double& maxVal) const
	{
		minVal = std::numeric_limits<double>::max();
		maxVal = -minVal;
		for (uint i1 = 0; i1 < dx * dy * dz; i1++)
		{
			double val = vol[i1].sdf;

			if (minVal > val) minVal = val;
			if (maxVal < val) maxVal = val;
		}
	}

	//! Set the voxel at i.
	inline void set(uint i, Voxel vox)
	{
		double val = vox.sdf;
		if (val > maxValue)
			maxValue = val;

		if (val < minValue)
			minValue = val;

		vol[i] = vox;
	};
	//! Set the value at i.
	inline void set(uint i, double val)
	{
		if (val > maxValue)
			maxValue = val;

		if (val < minValue)
			minValue = val;

		vol[i].sdf = val;
	};
	//! Set the voxel at (x_, y_, z_).
	inline void set(uint x_, uint y_, uint z_, Voxel vox)
	{
		vol[getPosFromTuple(x_, y_, z_)] = vox;
	};
	//! Set the value at (x_, y_, z_).
	inline void set(uint x_, uint y_, uint z_, double val)
	{
		vol[getPosFromTuple(x_, y_, z_)].sdf = val;
	};

	//!Get the voxel at i.
	inline Voxel getVoxel(uint i) const
	{
		return vol[i];
	};

	//! Get the value at i.
	inline double getValue(uint i) const
	{
		return vol[i].sdf;
	};
	//! Get the voxel at (x_, y_, z_).
	inline Voxel getVoxel(uint x_, uint y_, uint z_) const
	{
		return vol[getPosFromTuple(x_, y_, z_)];
	};

	//! Get the value at (x_, y_, z_).
	inline double getValue(uint x_, uint y_, uint z_) const
	{
		return vol[getPosFromTuple(x_, y_, z_)].sdf;
	};

	//! Get the value at (pos.x, pos.y, pos.z).
	inline double getValue(const Eigen::Vector3i& pos_) const
	{
		return(getValue(pos_[0], pos_[1], pos_[2]));
	}

	//! Returns the cartesian x-coordinates of node (i,..).
	inline double posX(int i) const
	{
		return min[0] + diag[0] * (double(i) * ddx);
	}

	//! Returns the cartesian y-coordinates of node (..,i,..).
	inline double posY(int i) const
	{
		return min[1] + diag[1] * (double(i) * ddy);
	}

	//! Returns the cartesian z-coordinates of node (..,i).
	inline double posZ(int i) const
	{
		return min[2] + diag[2] * (double(i) * ddz);
	}

	//! Returns the cartesian coordinates of node (i,j,k).
	inline Eigen::Vector3d pos(int i, int j, int k) const
	{
		Eigen::Vector3d coord(0, 0, 0);

		coord[0] = min[0] + (max[0] - min[0]) * (double(i) * ddx);
		coord[1] = min[1] + (max[1] - min[1]) * (double(j) * ddy);
		coord[2] = min[2] + (max[2] - min[2]) * (double(k) * ddz);

		return coord;
	}

	//! Returns number of cells in x-dir.
	inline uint getDimX() const { return dx; }

	//! Returns number of cells in y-dir.
	inline uint getDimY() const { return dy; }

	//! Returns number of cells in z-dir.
	inline uint getDimZ() const { return dz; }

	inline Eigen::Vector3d getMin() { return min; }
	inline Eigen::Vector3d getMax() { return max; }


	//! Get the index of the cell using the tuple (x,y,z) in the flattened vector.
	inline uint getPosFromTuple(int x, int y, int z) const
	{
		return x * dy * dz + y * dz + z;
	}

	//Set voxel at (x,y,z) to value
	inline void setVoxel(int x, int y, int z, Voxel value)
	{
		vol[getPosFromTuple(x, y, z)] = value;
	}

	// // Get volume as a 2d opencv matrix with dimensions (dy *dz, dx) and two channels (sdf, weight) 
	// cv::Mat getVolume() {
	// 	int sizes[3] = { static_cast<int>(dy * dz), static_cast<int>(dx), 2 };
	// 	cv::Mat volume = cv::Mat(3, sizes, CV_16SC1, cv::Scalar(0));
	// 	for (int i = 0; i < dx; i++) {
	// 		for (int j = 0; j < dy; j++) {
	// 			for (int k = 0; k < dz; k++) {
	// 				volume.at<short>(j * dz + k, i, 0) = vol_access(i, j, k).sdf;
	// 				volume.at<short>(j * dz + k, i, 1) = vol_access(i, j, k).weight;
	// 			}
	// 		}
	// 	}
	// 	return volume;
	// }
	
	// // Get color Volume as a 2d opencv matrix with dimensions (dy *dz, dx) and three channels (r, g, b) data type uchar
	// cv::Mat getColorVolume() {
	// 	int sizes[3] = { static_cast<int>(dy * dz), static_cast<int>(dx), 3 };
	// 	cv::Mat volume = cv::Mat(3, sizes, CV_8UC1, cv::Scalar(0));
	// 	for (int i = 0; i < dx; i++) {
	// 		for (int j = 0; j < dy; j++) {
	// 			for (int k = 0; k < dz; k++) {
	// 				volume.at<uchar>(j * dz + k, i, 0) = vol_access(i, j, k).color[0];
	// 				volume.at<uchar>(j * dz + k, i, 1) = vol_access(i, j, k).color[1];
	// 				volume.at<uchar>(j * dz + k, i, 2) = vol_access(i, j, k).color[2];
	// 			}
	// 		}
	// 	}
	// 	return volume;	
	// }

	// Get volume as a 2d opencv matrix with dimensions (dy *dz, dx) and two channels (sdf, weight) 
	cv::Mat getVolumeData() {
		int sizes[2] = { static_cast<int>(dy * dz), static_cast<int>(dx) };
		cv::Mat volume = cv::Mat(2, sizes, CV_16SC2);
		for (int i = 0; i < dx; i++) {
			for (int j = 0; j < dy; j++) {
				for (int k = 0; k < dz; k++) {
					volume.at<cv::Vec<short, 2>>(k * dy + j, i) = cv::Vec<short, 2>{static_cast<short>(vol_access(i, j, k).sdf), static_cast<short>(vol_access(i, j, k).weight)};
				}
			}
		}
		return volume;
	}
	
	// Get color Volume as a 2d opencv matrix with dimensions (dy *dz, dx) and three channels (r, g, b) data type uchar
	cv::Mat getColorVolumeData() {
		int sizes[2] = { static_cast<int>(dy * dz), static_cast<int>(dx) };
		cv::Mat volume = cv::Mat(2, sizes, CV_8UC3);
		for (int i = 0; i < dx; i++) {
			for (int j = 0; j < dy; j++) {
				for (int k = 0; k < dz; k++) {
					volume.at<cv::Vec3b>(k * dy + j, i) = cv::Vec3b{static_cast<uchar>(vol_access(i, j, k).color[0]), static_cast<uchar>(vol_access(i, j, k).color[1]), static_cast<uchar>(vol_access(i, j, k).color[2])};
				}
			}
		}
		return volume;	
	}

	cv::Mat getVolumeData_multi_threads() {
		int sizes[2] = { static_cast<int>(dy * dz), static_cast<int>(dx) };
		cv::Mat volume = cv::Mat(2, sizes, CV_16SC2);

		int numThreads = std::thread::hardware_concurrency();
		std::vector<std::thread> threads(numThreads);
		int zStep = dz / numThreads;

		for (int i = 0; i < numThreads; ++i) {
			int zStart = i * zStep;
			int zEnd = (i + 1) * zStep;
			if (i == numThreads - 1) {
				zEnd = dz;
			}
			auto boundFunc = std::bind(&Volume::getVolumeData_processVolumeSlice, this, std::ref(volume), dx, dy, dz, zStart, zEnd);
			threads[i] = std::thread(boundFunc);
		}

		for (auto& thread : threads) {
			thread.join();
		}

		return volume;
	}

	void getVolumeData_processVolumeSlice(cv::Mat& volume, int dx, int dy, int dz, int zStart, int zEnd) {
		for (int i = 0; i < dx; i++) {
			for (int j = 0; j < dy; j++) {
				for (int k = zStart; k < zEnd; k++) {
					volume.at<cv::Vec<short, 2>>(k * dy + j, i) = cv::Vec<short, 2>{static_cast<short>(vol_access(i, j, k).sdf), static_cast<short>(vol_access(i, j, k).weight)};
				}
			}
		}
	}

	cv::Mat getColorVolumeData_multi_threads() {
		int sizes[2] = { static_cast<int>(dy * dz), static_cast<int>(dx) };
		cv::Mat volume = cv::Mat(2, sizes, CV_8UC3);

		int numThreads = std::thread::hardware_concurrency();
		std::vector<std::thread> threads(numThreads);
		int zStep = dz / numThreads;

		for (int i = 0; i < numThreads; ++i) {
			int zStart = i * zStep;
			int zEnd = (i + 1) * zStep;
			if (i == numThreads - 1) {
				zEnd = dz;
			}
			auto boundFunc = std::bind(&Volume::getColorVolumeData_processColorVolumeSlice, this, std::ref(volume), dx, dy, dz, zStart, zEnd);
			threads[i] = std::thread(boundFunc);
		}

		for (auto& thread : threads) {
			thread.join();
		}

		return volume;
	}

	void getColorVolumeData_processColorVolumeSlice(cv::Mat& volume, int dx, int dy, int dz, int zStart, int zEnd) {
		for (int i = 0; i < dx; i++) {
			for (int j = 0; j < dy; j++) {
				for (int k = zStart; k < zEnd; k++) {
					volume.at<cv::Vec3b>(k * dy + j, i) = cv::Vec3b{
						static_cast<uchar>(vol_access(i, j, k).color[0]),
						static_cast<uchar>(vol_access(i, j, k).color[1]),
						static_cast<uchar>(vol_access(i, j, k).color[2])
					};
				}
			}
		}
	}

};

struct Frame {
	const CameraParameters camera_parameters;
	const unsigned char* color_map;
	const float* depth_map;
	const uint* class_map;

	std::vector<Eigen::Vector3f> verices;
	std::vector<Eigen::Vector3f> normals;
	Eigen::Matrix4f extrinsic;
	Eigen::Matrix4f depth_extrinsics;

	Frame(const CameraParameters camera_parameters, const unsigned char* color_map, const float* depth_map, const uint* class_map, Eigen::Matrix4f extrinsic, Eigen::Matrix4f depth_extrinsics) :
		camera_parameters(camera_parameters), color_map(color_map), depth_map(depth_map), class_map(class_map), extrinsic(extrinsic), depth_extrinsics(depth_extrinsics)
	{
	}

	Frame();


};

/*
    *
    * \brief Contains the internal volume representation
    *
    * This internal representation contains two volumes:
    * (1) TSDF volume: The global volume used for depth frame fusion and
    * (2) Color volume: Simple color averaging for colorized vertex output
    *
    * It also contains two important parameters:
    * (1) Volume size: The x, y and z dimensions of the volume (in mm)
    * (2) Voxel scale: The scale of a single voxel (in mm)
    *
    */
// -
struct VolumeData {
    cv::Mat tsdf_volume; //short2
    cv::Mat color_volume; //uchar4
	// Class data
	cv::Mat class_volume; //uint
    Eigen::Vector3i volume_size;
    float voxel_scale;

    VolumeData(const Eigen::Vector3i _volume_size, const float _voxel_scale) :
            //TSDF volume is 2 channel, one channel for TSDF value, one channel for weight
            tsdf_volume(cv::Mat(_volume_size[1] * _volume_size[2], _volume_size[0], CV_16SC2)),
            color_volume(cv::Mat(_volume_size[1] * _volume_size[2], _volume_size[0], CV_8UC3)),
			class_volume(cv::Mat(_volume_size[1] * _volume_size[2], _volume_size[0], CV_8UC1)),
            volume_size(_volume_size), voxel_scale(_voxel_scale)
    {
        // initialize the volume
        tsdf_volume.setTo(0);
        color_volume.setTo(0);
    }

	cv::Mat getTSDFVolume() {
		return tsdf_volume;
	}

	cv::Mat getColorVolume() {
		return color_volume;
	}

	cv::Mat getClassVolume() {
		return class_volume;
	}


	//! Returns the cartesian coordinates of node (row, col).
	inline Eigen::Vector3d pos(int row, int col) const
	{
		Eigen::Vector3d coord(0, 0, 0);

		int i = col;
		int k = row / volume_size[1];
		int j = row % volume_size[1];

		coord[0] = i * voxel_scale;
		coord[1] = j * voxel_scale;
		coord[2] = k * voxel_scale;

		return coord;
		}
};
} // namespace CPU