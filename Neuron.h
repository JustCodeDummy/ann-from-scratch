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

class Neuron {




	public:
		ActivationFunction activation;
		std::vector<double> weights, previousGradients, velocities;
		std::vector<Neuron *> inputs;
		double bias, output, gradient;

		int id_ = 0;

		explicit Neuron(ActivationFunction function, double bias, int id);

		void pushGradient(double gradient_);

		bool operator==(const Neuron &other) const;

		Neuron() : activation(SIGMOID), bias(0.0), id_(-1), output(0) {};
};