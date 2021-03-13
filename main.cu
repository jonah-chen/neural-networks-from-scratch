//
// Created by hina on 2021-03-11.
//
#include "matrix.cuh"

int main(int argc, char* argv[])
{
	try
	{
		Matrix m1(10000, 10000);
		Matrix m2(10000,10000);

		std::cout << m1;
		std::cout << m2;
		std::cout << m1 * m2;
		std::cout << m2 * m1;

		std::cout << m1.get_cpu();

		m2.to_cpu();
		std::cout << m2;
		std::cout << m2.get_gpu();

		// rectangular matrices
		Matrix m3 (2000,8000);
		Matrix m4 (8000,1000);
		std::cout << m3*m4;
		std::cout << m4*m3;
	}
	catch (char const* a)
	{
		std::cout << a;
	}
}