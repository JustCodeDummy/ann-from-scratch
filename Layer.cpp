//
// Created by fsociety on 03/09/2024.
//

#include "Layer.h"

DenseLayer::DenseLayer(int units, ActivationFunction activation, double bias) {
	for (int i = 0; i<units; i++){
		this->neurons.emplace_back(activation, bias, i+1);
	}
}

void Layer::connectNext(Layer *nextLayer) {
	for (auto &input: this->neurons){
		for (auto &neuron : nextLayer->neurons) {
			neuron.inputs.push_back(&input);
		}
	}
}






