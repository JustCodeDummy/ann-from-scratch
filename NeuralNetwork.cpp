#include "NeuralNetwork.h"
#include <cmath>
#include <random>
void NeuralNetwork::addLayer(Layer* layer) {
	this->layers.push_back(layer);
}

void NeuralNetwork::info() {
	// TODO display info
}
// initialize weights
void NeuralNetwork::compile() {

	LayerIterator iterator(this->layers);
	Layer* curr = iterator.getNext();

	while (iterator.hasNext()){
		Layer* next = iterator.getNext();
		curr->connectNext(next);
		curr = next;
	}

	for (auto & layer : this->layers){
		for (auto & n : layer->neurons){
			for (int _ = 0; _<layer->neurons.size(); _++){
				n.weights.push_back(NeuralNetwork::xavier_uniform(n.inputs.size(), layer->neurons.size()));
			}
		}
	}
}


double NeuralNetwork::xavier_uniform(long n_in, long n_out) {
	double limit = std::sqrt( 6.0 / (n_in + n_out));
	std::vector<double> vec;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-limit, limit);
	return dis(gen);
}
