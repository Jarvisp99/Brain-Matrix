#pragma once

#include "BMSetting.h"

namespace BM {

	// Addition operation changes the matrix through which it is called

	mat& mat::add(const mat& right) {
		cl::Context context({ defaultDevice });

		std::string kernelCode =
			"   __kernel void addMat(__global float* A, __global float* B){				"
			"       A[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];		"
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;
		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);
		cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * total, right.data);

		cl::Kernel kernel_add(program, "addMat");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, bufferB);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::finish();
		return *this;
	}

	//--------------------------------------------------------------

	// Subtraction operation changes the matrix through which it is called

	mat& mat::sub(const mat& right) {
		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void subMat(__global float* A, __global float* B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]-B[get_global_id(0)];    "
			"   }                                                                     ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;

		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);
		cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * total, right.data);

		cl::Kernel kernel_add(program, "subMat");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, bufferB);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::finish();
		return *this;
	}

	//-----------------------------------------------------------------------

	// Multiplication of two matrices

	mat& mat::operator*(const mat& right) {

		mat* temp = new mat(x, right.y);

		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void mulMat(__global float* A, __global float* B, __global float* C,int colA,int colB){        "
			"					float temp = 0;																		    "
			"					for(int i=0;i<colA;i++){															    "
			"						temp += A[(get_global_id(0)*colA)+i] * B[get_global_id(1) + (i*colB)];				"
			"					}																						"
			"					C[(get_global_id(0)*colB)+get_global_id(1)] = temp;										"
			"   }                                                                     ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int totalA = x * y;
		int totalB = right.x * right.y;

		if (y != right.x)
			return *this;

		int totalC = x * right.y;

		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * totalA);
		cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(float) * totalB);
		cl::Buffer bufferC(context, CL_MEM_READ_WRITE, sizeof(float) * totalC);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * totalA, data);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * totalB, right.data);

		cl::Kernel kernel_add(program, "mulMat");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, bufferB);
		kernel_add.setArg(2, bufferC);
		kernel_add.setArg(3, sizeof(float),(void*)&y);
		kernel_add.setArg(4, sizeof(float),(void*)&right.y);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(x,right.y), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * totalC, temp->data);

		cl::finish();
		return *temp;
	}

	//------------------------------------------------------------------------

	// Addition operation using '+' sign and creates a new matrix to pass

	mat& mat::operator+(const mat& right) {

		mat* temp = new mat(x, y);

		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void addMat(__global float* A, __global float* B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];       "
			"   }                                                                       ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;

		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);
		cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * total, right.data);

		cl::Kernel kernel_add(program, "addMat");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, bufferB);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, temp->data);

		cl::finish();
		return *temp;
	}

	//--------------------------------------------------------------

	// Subtraction operation using '-' sign and creates a new matrix to pass

	mat& mat::operator-(const mat& right) {

		mat* temp = new mat(x, y);

		cl::Context context({ defaultDevice });

		std::string kernelCode =
			"   __kernel void subMat(__global float* A, __global float* B){       "
			"       A[get_global_id(0)]=A[get_global_id(0)]-B[get_global_id(0)];                 "
			"   }                                                                               ";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		int total = x * y;

		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);
		cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * total, right.data);

		cl::Kernel kernel_add(program, "subMat");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, bufferB);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, temp->data);
		cl::finish();
		return *temp;
	}

	//----------------------------------------------------------------------------------------

}