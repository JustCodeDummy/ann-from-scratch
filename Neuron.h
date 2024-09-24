#pragma once
#include <vector>
#include <iostream>
#include <functional>


enum ActivationFunction {
	SIGMOID,
	BINARY,
	RELU,
	LINEAR,
	TANH
};

class Neuron{



	public:
		ActivationFunction activation;
		std::vector<double> weights;
		std::vector<Neuron*> inputs;
		double bias, output, gradient;

		explicit Neuron(ActivationFunction function, double bias);

};