#pragma once

/* This function is called at the beginning to initialise the heterogenous computing units available / connected to
		the system .
		The API used here is openCL 2.0 . And can change in future according to the needs and platform available. */



namespace BM {

	//GLOBAL SCOPED VARIABLES

	//These are vectorised for better experience when user have multiple GPUs
	//get all platforms (drivers)
	std::vector<cl::Platform> allPlatforms;

	//get all devices
	std::vector<cl::Device> allDevices;

	//Store the default Platform / Device data
	cl::Platform defaultPlatform;
	cl::Device defaultDevice;

	//-------------------------------------------
	void init() {

		cl::Platform::get(&allPlatforms);
		if (allPlatforms.size() == 0) {
			std::cout << " No platforms found. Check OpenCL installation!\n";
			exit(1);
		}
		defaultPlatform = allPlatforms[0];
		//std::cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";

		//get default device of the default platform
		defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
		if (allDevices.size() == 0) {
			std::cout << " No devices found. Check OpenCL installation!\n";
			exit(1);
		}
		defaultDevice = allDevices[0];
		//std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>() << "\n";

	}

}

//------------------------------------------------------------------------------------------------------------------