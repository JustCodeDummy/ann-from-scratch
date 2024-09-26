#include "Neuron.h"

#include <utility>

Neuron::Neuron(ActivationFunction function, double bias) {
	this->bias = bias;
	this->activation = function;
}

bool Neuron::operator==(const Neuron &other) const {
	return other.gradient == this->gradient &&
			other.output == this->output &&
			other.bias == this->bias;

}


