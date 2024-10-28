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


private:


	public:
		ActivationFunction activation;
		std::vector<double> weights, previousGradients, velocities;
		std::vector<Neuron*> inputs;
		double bias, output, gradient;

		int id_ = 0;
		explicit Neuron(ActivationFunction function, double bias, int id);

		void pushGradient(double gradient);

		bool operator==(const Neuron& other) const;


};