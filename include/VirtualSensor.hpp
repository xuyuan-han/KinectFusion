#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>


typedef unsigned char BYTE;

// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor {
public:

	VirtualSensor() : m_currentIdx(-1), m_increment(1) { }

	

	bool init(const std::string& datasetDir) {
		m_baseDir = datasetDir;

		// Read filename lists
		if (!readFileList(datasetDir + "depth.txt", m_filenameDepthImages, m_depthImagesTimeStamps)) return false;
		if (!readFileList(datasetDir + "rgb.txt", m_filenameColorImages, m_colorImagesTimeStamps)) return false;

		
		
		if (m_filenameDepthImages.size() != m_filenameColorImages.size()) return false;

		// Image resolutions
		m_colorImageWidth = 640;
		m_colorImageHeight = 480;
		m_depthImageWidth = 640;
		m_depthImageHeight = 480;

		// Intrinsics
		m_colorIntrinsics << 525.0f, 0.0f, 319.5f,
			0.0f, 525.0f, 239.5f,
			0.0f, 0.0f, 1.0f;

		m_depthIntrinsics = m_colorIntrinsics;

		m_colorExtrinsics.setIdentity();
		m_depthExtrinsics.setIdentity();

	

		m_currentIdx = -1;
		return true;
	}

	bool processNextFrame() {
		if (m_currentIdx == -1) m_currentIdx = 0;
		else m_currentIdx += m_increment;

		if ((unsigned int)m_currentIdx >= (unsigned int)m_filenameColorImages.size()) return false;

		std::cout << "ProcessNextFrame [" << m_currentIdx << " | " << m_filenameColorImages.size() << "]" << std::endl;

		cv::Mat rgbImage = cv::imread(m_baseDir + m_filenameColorImages[m_currentIdx]);
       

        // Depth images are scaled by 5000
        cv::Mat_<float> dImage = cv::imread(m_baseDir + m_filenameDepthImages[m_currentIdx], cv::IMREAD_UNCHANGED);

		dImage /= 5000.0f;

		return true;
	}

	unsigned int getCurrentFrameCnt() {
		return (unsigned int)m_currentIdx;
	}

	// get current color data
	cv::Mat getColorRGBX() {
		return rgbImage;
	}

	// get current depth data
	cv::Mat_<float> getDepth() {
		return dImage;
	}

	// color camera info
	Eigen::Matrix3f getColorIntrinsics() {
		return m_colorIntrinsics;
	}

	Eigen::Matrix4f getColorExtrinsics() {
		return m_colorExtrinsics;
	}

	unsigned int getColorImageWidth() {
		return m_colorImageWidth;
	}

	unsigned int getColorImageHeight() {
		return m_colorImageHeight;
	}

	// depth (ir) camera info
	Eigen::Matrix3f getDepthIntrinsics() {
		return m_depthIntrinsics;
	}

	Eigen::Matrix4f getDepthExtrinsics() {
		return m_depthExtrinsics;
	}

	unsigned int getDepthImageWidth() {
		return m_depthImageWidth;
	}

	unsigned int getDepthImageHeight() {
		return m_depthImageHeight;
	}

	// get current trajectory transformation
	

private:
	bool readFileList(const std::string& filename, std::vector<std::string>& result, std::vector<double>& timestamps) {
		std::ifstream fileDepthList(filename, std::ios::in);
		if (!fileDepthList.is_open()) return false;
		result.clear();
		timestamps.clear();
		std::string dump;
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		while (fileDepthList.good()) {
			double timestamp;
			fileDepthList >> timestamp;
			std::string filename;
			fileDepthList >> filename;
			if (filename == "") break;
			timestamps.push_back(timestamp);
			result.push_back(filename);
		}
		fileDepthList.close();
		return true;
	}

	

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// current frame index
	int m_currentIdx;

	int m_increment;

	// frame data
	cv::Mat rgbImage;
	cv::Mat_<float> dImage;


	// color camera info
	Eigen::Matrix3f m_colorIntrinsics;
	Eigen::Matrix4f m_colorExtrinsics;
	unsigned int m_colorImageWidth;
	unsigned int m_colorImageHeight;

	// depth (ir) camera info
	Eigen::Matrix3f m_depthIntrinsics;
	Eigen::Matrix4f m_depthExtrinsics;
	unsigned int m_depthImageWidth;
	unsigned int m_depthImageHeight;

	// base dir
	std::string m_baseDir;
	// filenamelist depth
	std::vector<std::string> m_filenameDepthImages;
	std::vector<double> m_depthImagesTimeStamps;
	// filenamelist color
	std::vector<std::string> m_filenameColorImages;
	std::vector<double> m_colorImagesTimeStamps;

	// trajectory
	std::vector<Eigen::Matrix4f> m_trajectory;
	std::vector<double> m_trajectoryTimeStamps;
};
