//
// Created by fsociety on 03/09/2024.
//

#include "Layer.h"



DenseLayer::DenseLayer(int units, const std::function<double(double)>& activation, double bias) {
	for (int i = 0; i<units; i++){
		this->neurons.emplace_back( activation, bias, HIDDEN);
	}
}

void Layer::connectNext(Layer *nextLayer) {
	for (auto n: this->neurons){
		for (auto & nn : nextLayer->neurons) {
			nn.inputs.push_back(&n);
		}
	}
}

InputLayer::InputLayer(int units, const std::function<double(double)>& activation, double bias) {
	for (int i = 0; i<units; i++){
		this->neurons.emplace_back(activation, bias, INPUT);
	}
}

OutputLayer::OutputLayer(int units, const std::function<double(double)> &activation, double bias) {
	for (int i = 0; i<units; i++){
		this->neurons.emplace_back(activation, bias, HIDDEN);
	}
}



