#pragma once
#include <vector>
#include <iostream>
#include <functional>

enum NeuronType {
	INPUT,
	OUTPUT,
	HIDDEN
};

class Neuron{

	public:
		std::function<double(double)> activation;
		std::vector<double> weights;
		std::vector<Neuron*> inputs; // Maybe you'll have to change it to std::vector<Neuron> !!!
		double bias, output, gradient;
		NeuronType type;
		explicit Neuron(std::function<double(double)> activation, double bias, NeuronType type);

		void addNeuron(Neuron* next){
			this->inputs.push_back(next);
		}
};