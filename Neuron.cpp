#include "Neuron.h"

#include <utility>

Neuron::Neuron(ActivationFunction function, double bias, int id) {
	this->bias = bias;
	this->activation = function;
	this->id_ = id;
}

bool Neuron::operator==(const Neuron &other) const {
	return other.gradient == this->gradient &&
			other.output == this->output &&
			other.bias == this->bias;

}


