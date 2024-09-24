#include "Neuron.h"

#include <utility>

Neuron::Neuron(ActivationFunction function, double bias) {
	this->bias = bias;
	this->activation = function;
}


