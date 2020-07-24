#include "ComputeMat.h"
#include "Core.h"
#include "MatAndVal.h"
#include "MatManip.h"

namespace BM {


	// Matrix obj's constructor (not default)

	mat::mat(int x, int y) {
		this->x = x;
		this->y = y;
		int total = x * y;
		data = (float*)malloc(total * sizeof(float));
	}

	//--------------------------------------------------------------

	// Fills matrix with a float value

	void mat::fill(const float& val) {
		cl::Context context({ defaultDevice });
		int total = x * y;
		cl::Buffer buffer(context, CL_MEM_READ_WRITE, sizeof(float) * total);
		cl::CommandQueue queue(context, defaultDevice);
		queue.enqueueFillBuffer(buffer, val, 0, sizeof(float) * total);
		queue.enqueueReadBuffer(buffer, CL_FALSE, 0, sizeof(float) * total, data);
		cl::finish();
	}

	//--------------------------------------------------------------------------------------------

	// Transposes the matrix

	mat& mat::transpose() {
		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode =
			"   __kernel void tranMat(__global float* A, __global float* B,int x,int y){	"
			"		int i = get_global_id(0);												"
			"		int row = i / y;														"
			"		int col = i % y;														"
			"		int id = col*x+row;														"
			"       B[id]=A[i];																"
			"   }																			";

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		auto err = program.build("-cl-std=CL1.2");

		int total = x * y;
		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);
		cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(float) * total);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "tranMat");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, bufferB);
		kernel_add.setArg(2, sizeof(int), (void *)&x);
		kernel_add.setArg(3, sizeof(int), (void *)&y);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(total), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(float) * total, data);

		int temp = x;
		x = y;
		y = temp;
		cl::finish();
		return *this;
	}

	//--------------------------------------------------------------------------------------------

	//	The function basically take one argument that is the axis and return sum axis - wise
	//	like axis = 0 means column wise and vise versa

	mat& mat::sum(const int& axis) {
		cl::Context context({ defaultDevice });

		// kernel calculates for each element C=A+B
		std::string kernelCode;
		int times;
		if (axis == 0) {
			times = x;
			kernelCode =
				"   __kernel void sumMat(__global float* A,__global float* B,int times,int cols){	"
				"		for(int i=0;i<times;i++)													"
				"			B[get_global_id(0)] += A[get_global_id(0) + (i*cols)];					"
				"   }																				";
		}
		else
		{
			times = y;
			kernelCode =
				"   __kernel void sumMat(__global float* A,__global float* B,int times,int cols){		"
				"		for(int i=0;i<times;i++)														"
				"			B[get_global_id(0)] += A[(get_global_id(0)*cols) + i];						"
				"   }																					";
		}

		cl::Program::Sources sources(1, std::make_pair(kernelCode.c_str(), kernelCode.length() + 1));

		cl::Program program(context, sources);
		auto err = program.build("-cl-std=CL1.2");

		int total = x * y;
		int size = (total / times);
		cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(float) * total);
		cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(float) * size);

		cl::CommandQueue queue(context, defaultDevice);

		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * total, data);

		cl::Kernel kernel_add(program, "sumMat");
		kernel_add.setArg(0, bufferA);
		kernel_add.setArg(1, bufferB);
		kernel_add.setArg(2, sizeof(int),(void*)&times);
		kernel_add.setArg(3, sizeof(int), (void*)&y);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(size), cl::NullRange);
		queue.finish();

		mat *res = new mat(size);
		queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(float) * size, res->data);
		cl::finish();
		return *res;
	}

	//----------------------------------------------------------------------------------------------


	// To extract data from the matrix for further calculations
	float& mat::operator() (int rows,int cols) {
		return data[rows * y + cols];
	}
	 
	//-----------------------------------------------------------------------------------------
	

	// Overloading -----(cout<<)----- function to output the matrix

	std::ostream& operator<< (std::ostream& stream, const mat& obj) {
		stream << "[\n";
		for (int i = 0; i < obj.x; i++) {
			stream << " ";
			for (int j = 0; j < obj.y; j++) {
				stream << obj.data[i*obj.y + j]<< " " ;
			}
			stream << "\n";
		}
		stream << " ]\n";
		return stream;
	}

	//-----------------------------------------------------------------


}