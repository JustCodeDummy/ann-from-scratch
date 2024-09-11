#include "Neuron.h"

#include <utility>


Neuron::Neuron(std::function<double(double)> activation, double bias, NeuronType type) {
	this->bias = bias;
	this->activation = std::move(activation);
	this->type = type;

}


