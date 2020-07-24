#pragma once
#include <iostream>

/*
										Brain - Matrix 

						This is main code base for the Brain Matrix Library

*/

namespace BM {

	class mat
	{
	private:
		float* data;
		int x, y;
	public:
		mat(int x, int y = 1);
		void fill(const float& val);

		//Other operations on matrices
		mat& transpose();

		mat& sum(const int& axis);

		//Operations on different matrix (Changes the value of the matrix)
		mat& add(const mat& right);
		mat& sub(const mat& right);

		//Operator overloading for add and subtract using + , - (Doesnot change the value of the matrix)
		mat& operator+(const mat& right);
		mat& operator-(const mat& right);
		mat& operator*(const mat& right);

		//Operations of matrix and a value (Changes the value of the matrix)
		mat& add(const float& right);
		mat& sub(const float& right);
		mat& mul(const float& right);

		//Operator overloading for add and subtract using + , - (Doesnot change the value of the matrix)
		mat& operator+(const float& right);
		mat& operator-(const float& right);
		mat& operator*(const float& right);


		//Overloaded the () functionality to change or operate on data at certain position
		float& operator() (int row,int cols = 1);


		// Overloaded the cout functionality
		friend std::ostream& operator<< (std::ostream& stream, const mat& obj);
	};

	void init();

}