#pragma once
#include <vector>
#include <iostream>
#include <functional>


enum ActivationFunction {
	SIGMOID,
	BINARY,
	RELU,
	LINEAR,
	TANH,
	SOFTMAX
};

class Neuron{



	public:
		ActivationFunction activation;
		std::vector<double> weights;
		std::vector<Neuron*> inputs;
		double bias, output, gradient;
		int id_ = 0;
		explicit Neuron(ActivationFunction function, double bias, int id);


		bool operator==(const Neuron& other) const;

};