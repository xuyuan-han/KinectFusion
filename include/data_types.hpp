#pragma once

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>


using Vector4uc = Eigen::Matrix<unsigned char, 4, 1>;

struct CameraParameters {
	int image_width, image_height;
	float focal_x, focal_y;
	float principal_x, principal_y;

	// Empty constructor
	CameraParameters() {}


	CameraParameters( const int image_width, const int image_height,
		const float focal_x, const float focal_y,
		const float principal_x, const float principal_y) :
		image_width(image_width), image_height(image_height),
		focal_x(focal_x), focal_y(focal_y),
		principal_x(principal_x), principal_y(principal_y)
	{ }

	CameraParameters(
		const int image_width, const int image_height,
		Eigen::Matrix3f K)
	{ 
		this->image_width = image_width;
		this->image_height = image_height;
		this->focal_x = K(0, 0);
		this->focal_y = K(1, 1);
		this->principal_x = K(0, 2);
		this->principal_y = K(1, 2);
	}
	/**
 * Returns camera parameters for a specified pyramid level; each level corresponds to a scaling of pow(.5, level)
 * @param level The pyramid level to get the parameters for with 0 being the non-scaled version,
 * higher levels correspond to smaller spatial size
 * @return A CameraParameters structure containing the scaled values
 */
	CameraParameters level(const size_t level) const
	{
		if (level == 0) return *this;

        const float scale_factor = powf(0.5f, static_cast<float>(level));
        return CameraParameters { image_width >> level, image_height >> level,
                                    focal_x * scale_factor, focal_y * scale_factor,
                                    (principal_x + 0.5f) * scale_factor - 0.5f,
                                    (principal_y + 0.5f) * scale_factor - 0.5f };
    }


	Eigen::Matrix3f getIntrinsicMatrix() const
	{
		Eigen::Matrix3f K;
		K << focal_x, 0, principal_x,
			0, focal_y, principal_y,
			0, 0, 1;
		return K;
	}
};

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
	void Volume::zeroOutMemory()
	{
		for (uint i1 = 0; i1 < dx * dy * dz; i1++)
			vol[i1].sdf = double(0.0);
	}

	//! Returns the Data.
	Voxel* Volume::getData()
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

	// Get volume as a opencv matrix
	cv::Mat getVolume() {
		int sizes[3] = { dx, dy, dz };
		cv::Mat volume = cv::Mat(3, sizes, CV_32FC1, cv::Scalar(0));
		for (int i = 0; i < dx; i++) {
			for (int j = 0; j < dy; j++) {
				for (int k = 0; k < dz; k++) {
					volume.at<float>(i, j, k) = vol_access(i, j, k).sdf;
				}
			}
		}
	}
	
	// Get weight as a opencv matrix
	cv::Mat getWeight() {
			int sizes[3] = { dx, dy, dz };
			cv::Mat weight = cv::Mat(3, sizes, CV_32FC1, cv::Scalar(0));
			for (int i = 0; i < dx; i++) {
				for (int j = 0; j < dy; j++) {
					for (int k = 0; k < dz; k++) {
						weight.at<float>(i, j, k) = vol_access(i, j, k).weight;
					}
				}
			}
		}  

};

struct Frame {
	 CameraParameters camera_parameters;
	 unsigned char* color_map;
	 float* depth_map;
	 uint* class_map;

	std::vector<Eigen::Vector3f> verices;
	std::vector<Eigen::Vector3f> normals;
	Eigen::Matrix4f extrinsic;
	Eigen::Matrix4f depth_extrinsics;

	// Empty constructor
	Frame() {}

	Frame(const CameraParameters camera_parameters,  unsigned char* color_map,  float* depth_map,  uint* class_map, Eigen::Matrix4f extrinsic, Eigen::Matrix4f depth_extrinsics) :
		camera_parameters(camera_parameters), color_map(color_map), depth_map(depth_map), class_map(class_map), extrinsic(extrinsic), depth_extrinsics(depth_extrinsics)
	{
	}

	

};
