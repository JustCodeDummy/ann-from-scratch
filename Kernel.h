#pragma once

#include <vector>
#include <functional>
#include <cmath>
class Kernel {
	private:

		std::vector<std::vector<double>> kernel;
		size_t rows, cols;
	public:

		explicit Kernel(size_t rows, size_t cols, std::function<double(int, int)>& initialize);
		Kernel();

		Kernel(std::vector<std::vector<double>> kernel_){this->kernel = kernel_;}

		Kernel(size_t dim){this->kernel = std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0));}

		double initializer(int, int){
			return 0.0;
		};

		size_t size();

		std::vector<double>& operator[](size_t row);

		const std::vector<double>& operator[](size_t row) const;

		Kernel operator *(double scalar);

		void operator -=(Kernel& matrix);

		void operator+=(Kernel& matrix);

		std::vector<std::vector<double>> toVector(){return this->kernel;}

};


