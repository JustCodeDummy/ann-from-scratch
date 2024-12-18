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

void Neuron::pushGradient(double gradient_) {
	this->previousGradients.push_back(gradient_);
	if (this->previousGradients.size() > 100){
		this->previousGradients.erase(this->previousGradients.begin());
	}
}

std::vector<std::vector<double>> Neuron::to2D(int kSize) {

	auto inputRegion = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize, 0));

	for (int i; i<kSize; i++){
		for (int j =0; j<kSize; j++){
			inputRegion[i][j] = this->inputs[j + i*kSize]->output;
		}
	}
}




