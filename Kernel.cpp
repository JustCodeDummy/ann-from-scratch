#include <stdexcept>
#include "Kernel.h"

Kernel::Kernel(size_t rows, size_t cols, std::function<double(int, int)>& initialize) {
	this->kernel = std::vector<std::vector<double>>(rows, std::vector<double>(cols, initialize(rows, cols)));
	this->rows = rows;
	this->cols = cols;
}

Kernel::Kernel(){
	this->kernel = std::vector<std::vector<double>>(3, std::vector<double>(cols, 0));
	this->rows = 3;
	this->cols = 3;
}

size_t Kernel::size() {
	return (size_t) this->kernel.size();
}

std::vector<double> &Kernel::operator[](size_t row) {
	if (row >= rows) {
		throw std::out_of_range("Row index out of range");
	}
	return this->kernel[row];
}

const std::vector<double> &Kernel::operator[](size_t row) const {
	if (row >= rows) {
		throw std::out_of_range("Row index out of range");
	}
	return this->kernel[row];

}

Kernel Kernel::operator*(double scalar) {
	size_t s = this->size();
	Kernel kernel_ = Kernel(std::vector<std::vector<double>>(s, std::vector<double>(s, 0)));
	for (size_t i=0; i<s; i++){
		for (size_t j=0; j<s; j++){
			kernel_[i][j] = this->kernel[i][j] * scalar;
		}
	}
	return kernel_;
}

void Kernel::operator-=(Kernel& matrix){
	for (size_t i = 0; i<matrix.size(); i++){
		for (size_t j = 0; j<matrix[0].size(); i++){
			kernel[i][j] -= matrix[i][j];
		}
	}
}

void Kernel::operator+=(Kernel& matrix){
	for (size_t i = 0; i<matrix.size(); i++){
		for (size_t j = 0; j<matrix[0].size(); i++){
			kernel[i][j] += matrix[i][j];
		}
	}
}


