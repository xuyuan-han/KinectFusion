#include "surface_reconstruction.hpp"

// multi thread version
void Surface_Reconstruction::integrate_multi_threads(cv::Mat depth, cv::Mat colorMap, Volume* vol, CameraParameters camera_parameters, float trancutionDistance, Eigen::Matrix4f pos) {
    Eigen::Matrix4f worldToCamera = pos;
    Eigen::Matrix4f cameraToWorld = worldToCamera.inverse();
    Eigen::Matrix3f intrinsics = camera_parameters.getIntrinsicMatrix();
    int width = camera_parameters.image_width;
    int height = camera_parameters.image_height;

    float* depth_map = depth.ptr<float>();

    uint* class_map = new uint[width * height];

    int numThreads = std::thread::hardware_concurrency();
	// std::cout << "Number of threads: " << numThreads << std::endl;
    std::vector<std::thread> threads(numThreads);

    int zStep = vol->getDimZ() / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int zStart = i * zStep;
        int zEnd = (i + 1) * zStep;
        if (i == numThreads - 1) {
            zEnd = vol->getDimZ();
        }
        threads[i] = std::thread(Surface_Reconstruction::reconstructionProcessVolumeSlice, vol, colorMap, depth_map, class_map, cameraToWorld, intrinsics, width, height, trancutionDistance, zStart, zEnd);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    delete[] class_map;
}

void Surface_Reconstruction::reconstructionProcessVolumeSlice(Volume* vol, cv::Mat colorMap, float* depth_map, uint* class_map, Eigen::Matrix4f cameraToWorld, Eigen::Matrix3f intrinsics, int width, int height, float trancutionDistance, int zStart, int zEnd) {
    for (int z = zStart; z < zEnd; z++) {
        for (int y = 0; y < vol->getDimY(); y++) {
            for (int x = 0; x < vol->getDimX(); x++) {
				// Indices to world coordinates
				Eigen::Vector3d worldPoint = vol->pos(x, y, z);

				// To Homogeneous coordinates
				Eigen::Vector4f worldPointH = Eigen::Vector4f(worldPoint[0], worldPoint[1], worldPoint[2], 1);
				// To camera frame coordinates
				Eigen::Vector4f cameraPointH = cameraToWorld * worldPointH;
				// Non Homogeneous coordinates
				Eigen::Vector3f cameraPointNonHomogenous = Eigen::Vector3f(cameraPointH[0] / cameraPointH[3], cameraPointH[1] / cameraPointH[3], cameraPointH[2] / cameraPointH[3]);
				// To Sensor coordinates by projection
				Eigen::Vector3f cameraPoint = intrinsics * Eigen::Vector3f(cameraPointH[0] / cameraPointH[3], cameraPointH[1] / cameraPointH[3], cameraPointH[2] / cameraPointH[3]);
				// To pixel coordinates
				Eigen::Vector2i pixel = Eigen::Vector2i((int)round(cameraPoint[0] / cameraPoint[2]), (int)round(cameraPoint[1] / cameraPoint[2]));

				if (pixel[0] >= 0 && pixel[0] < width && pixel[1] >= 0 && pixel[1] < height)
				{
					float depth = depth_map[pixel[1] * width + pixel[0]];
					if (depth > 0)
					{
						//Calculate Lambda
						double lambda = getLambda(pixel, intrinsics);
						double sdf = (-1.f) * ((1.0f / lambda) * cameraPointNonHomogenous.norm() - depth);

						if (sdf >= -trancutionDistance)
						{
							float truncatedSDF = fmin(trancutionDistance, sdf);
							float weight = 1.0f;
							float oldSdf = vol->getVoxel(x, y, z).sdf;
							float oldWeight = vol->getVoxel(x, y, z).weight;
							uint oldClass = vol->getVoxel(x, y, z).class_id;

							float newSdf = (oldSdf * oldWeight + truncatedSDF * weight) / (oldWeight + weight);
							float newWeight = oldWeight + weight;

							uint newClass = class_map[pixel[1] * width + pixel[0]];

							if (newClass != oldClass)
							{
								// Randomly choose one of the classes with probability proportional to the weight
								double r = ((double)rand() / (RAND_MAX)); // Random number between 0 and 1
								if (r < (double)oldWeight / (oldWeight + weight))
									newClass = oldClass;
							}

							Vector4uc oldColor = vol->getVoxel(x, y, z).color;
							Vector4uc color = oldColor;

							if (sdf <= trancutionDistance / 2 && sdf >= -trancutionDistance / 2) {
								// color[0] = (color[0] * oldWeight + colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[0] * weight) / (oldWeight + weight);
								// color[1] = (color[1] * oldWeight + colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[1] * weight) / (oldWeight + weight);
								// color[2] = (color[2] * oldWeight + colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[2] * weight) / (oldWeight + weight);

								color[0] = colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[0];
								color[1] = colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[1];
								color[2] = colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[2];

								// for now just use the color of the voxel
								// color[0] = 255;
								// color[1] = 255;
								// color[2] = 255;
								color[3] = 255;
							}

							Voxel newVoxel = Voxel();
							newVoxel.sdf = newSdf;
							newVoxel.weight = newWeight;
							newVoxel.class_id = newClass;
							newVoxel.color = color;


							vol->setVoxel(x, y, z, newVoxel);

						}


					}
				}
            }
        }
    }
}

void Surface_Reconstruction::surface_reconstruction(cv::Mat depth, cv::Mat colorMap, VolumeData vol, float trancutionDistance, Eigen::Matrix4f pos)
{
	Eigen::Matrix4f worldToCamera = pos;
	Eigen::Matrix4f cameraToWorld = worldToCamera.inverse();
	Eigen::Matrix3f intrinsics = Eigen::Matrix3f::Identity();
	Eigen::Matrix3f intrinsicsInv = intrinsics.inverse();
	int width = depth.cols;
	int height = depth.rows;

	Eigen::Matrix3d R = worldToCamera.block<3, 3>(0, 0).cast<double>();
	Eigen::Vector3d t = worldToCamera.block<3, 1>(0, 3).cast<double>();

	for (int x = 0; x < vol.volume_size[2]; x++)
		for (int y = 0; y < vol.volume_size[1]; y++)
			for (int z = 0; y < vol.volume_size[0]; z++) {
				// Get point in the word frame. 


			};

}

double Surface_Reconstruction::getLambda(Eigen::Vector2i pixel, Eigen::Matrix3f intrinsics)
{
	double fovX = intrinsics(0, 0);
	double fovY = intrinsics(1, 1);
	double cX = intrinsics(0, 2);
	double cY = intrinsics(1, 2);
	const Eigen::Vector3d lambda(
		(pixel.x() - cX) / fovX,
		(pixel.y() - cY) / fovY,
		1.);

	return lambda.norm();
};

// single thread version
void Surface_Reconstruction::integrate(cv::Mat depth, cv::Mat colorMap, Volume* vol,CameraParameters camera_parameters , float trancutionDistance, Eigen::Matrix4f pos)
{


	Eigen::Matrix4f worldToCamera= pos;
	Eigen::Matrix4f cameraToWorld= worldToCamera.inverse();
	Eigen::Matrix3f intrinsics= camera_parameters.getIntrinsicMatrix();
	Eigen::Matrix3f intrinsicsInv= intrinsics.inverse();
	int width= camera_parameters.image_width;
	int height= camera_parameters.image_height;
	
	Eigen:: Matrix3d R= cameraToWorld.block<3,3>(0,0).cast<double>();
	Eigen::Vector3d t= cameraToWorld.block<3,1>(0,3).cast<double>();


	// Convert depth map to float *
	float* depth_map = depth.ptr<float>();


	// Dummy class map for now
	uint * class_map = new uint[width * height];



	for (int z = 0; z < vol->getDimZ(); z++)
		for (int y = 0; y < vol->getDimY(); y++)
			for (int x = 0; x < vol->getDimX(); x++)
			{
				// Indices to world coordinates
				Eigen::Vector3d worldPoint = vol->pos(x, y, z);
				// To Homogeneous coordinates
				Eigen::Vector4f worldPointH = Eigen::Vector4f(worldPoint[0], worldPoint[1], worldPoint[2], 1);
				// To camera frame coordinates
				Eigen::Vector4f cameraPointH = cameraToWorld * worldPointH;
				// Non Homogeneous coordinates
				Eigen::Vector3f cameraPointNonHomogenous = Eigen::Vector3f(cameraPointH[0] / cameraPointH[3], cameraPointH[1] / cameraPointH[3], cameraPointH[2] / cameraPointH[3]);
				// To Sensor coordinates by projection
				Eigen::Vector3f cameraPoint = intrinsics * Eigen::Vector3f(cameraPointH[0] / cameraPointH[3], cameraPointH[1] / cameraPointH[3], cameraPointH[2] / cameraPointH[3]);
				// To pixel coordinates
				Eigen::Vector2i pixel = Eigen::Vector2i((int)round(cameraPoint[0] / cameraPoint[2]), (int)round(cameraPoint[1] / cameraPoint[2]));

				if (pixel[0] >= 0 && pixel[0] < width && pixel[1] >= 0 && pixel[1] < height)
				{
					float depth = depth_map[pixel[1] * width + pixel[0]];
					if (depth > 0)
					{
						//Calculate Lambda
						double lambda = getLambda(pixel, intrinsics);
						double sdf = (-1.f) * ((1.0f / lambda) * cameraPointNonHomogenous.norm() - depth);

						if (sdf >= -trancutionDistance)
						{
							float truncatedSDF = fmin(trancutionDistance, sdf);
							float weight = 1.0f;
							float oldSdf = vol->getVoxel(x, y, z).sdf;
							float oldWeight = vol->getVoxel(x, y, z).weight;
							uint oldClass = vol->getVoxel(x, y, z).class_id;

							float newSdf = (oldSdf * oldWeight + truncatedSDF * weight) / (oldWeight + weight);
							float newWeight = oldWeight + weight;

							uint newClass = class_map[pixel[1] * width + pixel[0]];

							if (newClass != oldClass)
							{
								// Randomly choose one of the classes with probability proportional to the weight
								double r = ((double)rand() / (RAND_MAX)); // Random number between 0 and 1
								if (r < (double)oldWeight / (oldWeight + weight))
									newClass = oldClass;
							}
							Vector4uc oldColor = vol->getVoxel(x, y, z).color;
							Vector4uc color = Vector4uc();

							if (sdf <= trancutionDistance / 2 && sdf >= -trancutionDistance / 2) {
								// color[0] = (color[0] * oldWeight + colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[0] * weight) / (oldWeight + weight);
								// color[1] = (color[1] * oldWeight + colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[1] * weight) / (oldWeight + weight);
								// color[2] = (color[2] * oldWeight + colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[2] * weight) / (oldWeight + weight);

								color[0] = colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[0];
								color[1] = colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[1];
								color[2] = colorMap.at<cv::Vec3b>(pixel[1], pixel[0])[2];

								// for now just use the color of the voxel
								// color[0] = 255;
								// color[1] = 255;
								// color[2] = 255;
								color[3] = 255;
							}

							Voxel newVoxel = Voxel();
							newVoxel.sdf = newSdf;
							newVoxel.weight = newWeight;
							newVoxel.class_id = newClass;
							newVoxel.color = color;


							vol->setVoxel(x, y, z, newVoxel);

						}


					}
				}
			};




}