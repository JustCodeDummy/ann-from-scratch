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
		curr->connectNext(iterator.getNext());
	}


}


double NeuralNetwork::xavier_uniform(int n_in, int n_out) {
	double limit = std::sqrt( 6.0 / (n_in + n_out));
	std::vector<double> vec;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-limit, limit);
	return dis(gen);
}
