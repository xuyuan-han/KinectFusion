#pragma once
#ifndef SURFACE_RECON_HPP
#define SURFACE_RECON_HPP

#include "data_types.hpp"

namespace Surface_Reconstruction {
	//! Integrate a new reading into the TSDF volume
	//! First parameter is the TSDF volume
	//! Second parameter is the Frame to integrate
	void integrate(Volume vol, Frame frame);

}

#endif // !SURFACE_RECON_HPP