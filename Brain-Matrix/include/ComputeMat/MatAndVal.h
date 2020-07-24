#pragma once

/*

												Brain - Matrix 

						This header file contains the matrix and value related operation codes

*/

#include "BMSetting.h"

namespace BM {


	// Contains Matrix and value related calculations

	mat& mat::add(const float& right) {
		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void addMatAndVal(__global float* A, float B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]+B;       "
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;
		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "addMatAndVal");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, sizeof(float), (void*)&right);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::finish();
		return *this;
	}

	//------------------------------------------------------------------------------------

	//	Substraction of a matrix and a value

	mat& mat::sub(const float& right) {
		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void subMatAndVal(__global float* A, float B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]-B;						"
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;
		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "subMatAndVal");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, sizeof(float), (void*)&right);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::finish();
		return *this;
	}

	//------------------------------------------------------------------------------------

	//	Multiplication of a matrix and a value

	mat& mat::mul(const float& right) {
		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void mulMatAndVal(__global float* A, float B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]*B;						"
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;
		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "mulMatAndVal");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, sizeof(float), (void*)&right);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::finish();
		return *this;
	}

	//------------------------------------------------------------------------------------

	// Addition operation using '+' sign and creates a new matrix to pass and it is a matrix and value operations

	mat& mat::operator+(const float& right) {
		mat* temp = new mat(x, y);

		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void addMatAndVal(__global float* A, float B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]+B;					"
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;

		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "addMatAndVal");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, sizeof(float), (void*)&right);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, temp->data);

		cl::finish();
		return *temp;
	}

	//----------------------------------------------------------------------------------------

	// Substraction operation using '-' sign and creates a new matrix to pass and it is a matrix and value operations

	mat& mat::operator-(const float& right) {
		mat* temp = new mat(x, y);

		cl::Context context({ defaultDevice });

		std::string kernelCode =
			"   __kernel void subMatAndVal(__global float* A, float B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]-B;					"
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;

		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "subMatAndVal");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, sizeof(float), (void*)&right);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, temp->data);

		cl::finish();
		return *temp;
	}

	//----------------------------------------------------------------------------------------

	// Multiplication operation using '*' sign and creates a new matrix to pass and it is a matrix and value operations

	mat& mat::operator*(const float& right) {
		mat* temp = new mat(x, y);

		cl::Context context({ defaultDevice });

		std::string kernelCode =
			"   __kernel void mulMatAndVal(__global float* A, float B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]*B;					"
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;

		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "mulMatAndVal");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, sizeof(float), (void*)&right);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, temp->data);

		cl::finish();
		return *temp;
	}

	//----------------------------------------------------------------------------------------

	//----------------------------------------------------------------------------------------
}