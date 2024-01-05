#include "surface_reconstruction.hpp"

void Surface_Reconstruction::integrate(Volume vol, Frame frame)
{
	Eigen::Matrix4f worldToCamera= frame.extrinsic;
	Eigen::Matrix4f cameraToWorld= worldToCamera.inverse();
	Eigen::Matrix3f intrinsics= frame.camera_parameters.getIntrinsicMatrix();
	Eigen::Matrix3f intrinsicsInv= intrinsics.inverse();
	int width= frame.camera_parameters.image_width;
	int height= frame.camera_parameters.image_height;
	
	Eigen:: Matrix3d R= worldToCamera.block<3,3>(0,0).cast<double>();
	Eigen:: Vector3d t= worldToCamera.block<3,1>(0,3).cast<double>();
	for (int z = 0; z<vol.getDimZ(); z++)
		for (int y = 0; y<vol.getDimY(); y++)
			for (int x = 0; x<vol.getDimX(); x++)
			{
				// Indices to world coordinates
				Eigen::Vector3d worldPoint= vol.pos(x,y,z);
				// To Homogeneous coordinates
				Eigen::Vector4f worldPointH= Eigen::Vector4f(worldPoint[0],worldPoint[1],worldPoint[2],1);
				// To camera frame coordinates
				Eigen::Vector4f cameraPointH= worldToCamera*worldPointH;
				// To Sensor coordinates by projection
				Eigen::Vector3f cameraPoint= intrinsics*Eigen::Vector3f(cameraPointH[0]/cameraPointH[3],cameraPointH[1]/cameraPointH[3],cameraPointH[2]/cameraPointH[3]);
				// To pixel coordinates
				Eigen::Vector2i pixel= Eigen::Vector2i((int)round(cameraPoint[0]/cameraPoint[2]), (int)round(cameraPoint[1]/cameraPoint[2]));

				if (pixel[0]>=0 && pixel[0]<width && pixel[1]>=0 && pixel[1]<height)
				{
					float depth= frame.depth_map[pixel[1]*width+pixel[0]];
					if (depth>0)
					{
						
					}
				}
			};




}
