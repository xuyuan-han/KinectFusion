#include "surface_reconstruction.hpp"

void Surface_Reconstruction::integrate(Volume* vol, Frame frame, float truncationDistance)
{
	Eigen::Matrix4f worldToCamera= frame.extrinsic;
	Eigen::Matrix4f cameraToWorld= worldToCamera.inverse();
	Eigen::Matrix3f intrinsics= frame.camera_parameters.getIntrinsicMatrix();
	Eigen::Matrix3f intrinsicsInv= intrinsics.inverse();
	int width= frame.camera_parameters.image_width;
	int height= frame.camera_parameters.image_height;
	
	Eigen:: Matrix3d R= worldToCamera.block<3,3>(0,0).cast<double>();
	Eigen:: Vector3d t= worldToCamera.block<3,1>(0,3).cast<double>();
	for (int z = 0; z < vol->getDimZ(); z++)
		for (int y = 0; y < vol->getDimY(); y++)
			for (int x = 0; x < vol->getDimX(); x++)
			{
				// Indices to world coordinates
				Eigen::Vector3d worldPoint = vol->pos(x, y, z);
				// To camera frame coordinates
				Eigen::Vector3f cameraPointReal = R.cast<float>() * worldPoint.cast<float>() - t.cast<float>();

				Eigen::Vector4f cameraPointH = Eigen::Vector4f(cameraPointReal[0], cameraPointReal[1], cameraPointReal[2], 1.0f);
				// Non Homogeneous coordinates
				Eigen::Vector3f cameraPointNonHomogenous = Eigen::Vector3f(cameraPointH[0] / cameraPointH[3], cameraPointH[1] / cameraPointH[3], cameraPointH[2] / cameraPointH[3]);
				// To Sensor coordinates by projection
				Eigen::Vector3f cameraPoint = intrinsics * Eigen::Vector3f(cameraPointH[0] / cameraPointH[3], cameraPointH[1] / cameraPointH[3], cameraPointH[2] / cameraPointH[3]);
				// To pixel coordinates
				Eigen::Vector2i pixel = Eigen::Vector2i((int)round(cameraPoint[0] / cameraPoint[2]), (int)round(cameraPoint[1] / cameraPoint[2]));


				if (pixel[0] >= 0 && pixel[0] < width && pixel[1] >= 0 && pixel[1] < height)
				{
					printf("pixel: %d %d\n", pixel[0], pixel[1]);
					printf("cameraPointNonHomogenous: %f %f %f\n", cameraPointNonHomogenous[0], cameraPointNonHomogenous[1], cameraPointNonHomogenous[2]);

					

					float depth = frame.depth_map[pixel[1] * width + pixel[0]];
					if (depth > 0)
					{
						//Calculate Lambda
						double lambda = getLambda(pixel, intrinsics);
						double sdf = (-1.f) * ((1.0f / lambda) * cameraPointNonHomogenous.norm() - depth);
						if (sdf >= -truncationDistance)
						{
							float weight = 1.0f;
							float oldSdf = vol->getVoxel(x, y, z).sdf;
							float oldWeight = vol->getVoxel(x, y, z).weight;
							uint oldClass = vol->getVoxel(x, y, z).class_id;

							float newSdf = (oldSdf * oldWeight + sdf * weight) / (oldWeight + weight);
							float newWeight = oldWeight + weight;

							uint newClass = frame.class_map[pixel[1] * width + pixel[0]];

							if (newClass != oldClass)
							{
								// Randomly choose one of the classes with probability proportional to the weight
								double r = ((double)rand() / (RAND_MAX)); // Random number between 0 and 1
								if (r < (double)oldWeight / (oldWeight + weight))
									newClass = oldClass;
							}
							Vector4uc oldColor = vol->getVoxel(x, y, z).color;
							Vector4uc color = Vector4uc();
							color[0] = (frame.color_map[3 * (pixel[1] * width + pixel[0])] * weight + oldColor[0] * oldWeight) / (oldWeight + weight);
							color[1] = (frame.color_map[3 * (pixel[1] * width + pixel[0]) + 1] * weight + oldColor[1] * oldWeight) / (oldWeight + weight);
							color[2] = (frame.color_map[3 * (pixel[1] * width + pixel[0]) + 2] * weight + oldColor[2] * oldWeight) / (oldWeight + weight);
							color[3] = 255;
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
