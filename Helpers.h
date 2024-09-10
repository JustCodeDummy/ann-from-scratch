#pragma once

#include <utility>
#include <vector>
#include <cstdint>
/**
  TODO When done with functionalities, change DataMatrix to Koalas DataFrame
 */

class DataMatrix {
	public:
		std::vector<std::vector<double>> matrix;
		unsigned long shape[2];

		explicit DataMatrix(std::vector<std::vector<double>> matrix){
			this->shape[0] = matrix.size();
			this->shape[1] = matrix[0].size();
			this->matrix = std::move(matrix);
		}
};

