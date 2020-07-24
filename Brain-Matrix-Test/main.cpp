#include <ComputeMat/ComputeMat.h>
#include <iostream>

//using namespace BM;

int main() {

	BM::init();
	int row = 5;
	int col = 10;
	
	BM::mat O1(row, col);

	O1.fill(5.2);
	O1.sub(7.6);
	std::cout << O1;
	
	//Access Position of mat
	std::cout<<O1(3, 2)<<"\n";

	BM::mat O2(row, col);

	O2.fill(4.5);
	BM::mat O3 = O2 * 1.2;

	std::cout << O3;

	BM::mat data1(3, 4), data2(4, 9);
	data1.fill(1);
	data2.fill(2);
	data1(0, 0) = 0;

	//Matrix Mul
	std::cout << data1 * data2;
}