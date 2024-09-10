
#include <cmath>
#include "ActivationFunctions.h"

namespace ActivationFunctions{

	double Sigmoid(double x){
		return 1 / (1 + std::exp(-x));
	}

	double Binary(double x){
		return x > 0 ? 1 : 0;
	}

	double ReLU(double x){
		return x > 0 ? x : 0;
	}

	double Linear(double x){
		return x;
	}

	double Tanh(double x){
		return std::tanh(x);
	}
}