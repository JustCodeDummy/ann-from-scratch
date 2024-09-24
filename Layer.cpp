//
// Created by fsociety on 03/09/2024.
//

#include "Layer.h"

DenseLayer::DenseLayer(int units, ActivationFunction activation, double bias) {
	for (int i = 0; i<units; i++){
		this->neurons.emplace_back( activation, bias);
	}
}

void Layer::connectNext(Layer *nextLayer) {
	for (auto n: this->neurons){
		for (auto & nn : nextLayer->neurons) {
			nn.inputs.push_back(&n);
		}
	}
}






