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
				n.weights.push_back(NeuralNetwork::xavier_uniform((long) n.inputs.size(), (long) layer->neurons.size()));
			}
		}
	}
}
double Sigmoid(double x){
	return 1 / (1 + std::exp(-x));
}

double Binary(double x){
	return x > 0 ? 1 : 0;
}

double ReLU(double x){
	return x > 0 ? x : 0;
}

double Linear(double x){
	return x;
}

double Tanh(double x){
	return std::tanh(x);
}

double NeuralNetwork::xavier_uniform(long n_in, long n_out) {
	double limit = std::sqrt( 6.0 / (n_in + n_out));
	std::vector<double> vec;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-limit, limit);
	return dis(gen);
}

double evaluate(const double x, ActivationFunction condition){
	switch (condition) {
		case SIGMOID:
			return Sigmoid(x);
		case BINARY:
			return Binary(x);
		case RELU:
			return ReLU(x);
		case LINEAR:
			return x;
		case TANH:
			return Tanh(x);
	}
}

double outputGradient(ActivationFunction condition, double x, double d){
	 // Currently only MSE error is implemented
	 // X is value, D is expected value

	double o;
	switch (condition) {
		case SIGMOID:
			 o = Sigmoid(x);
			// o * (1-o) is sigmoid derivative, (d - o) is MSE derivative
			return o * (1-o) * (d - o);
		case BINARY:
			break;
		case RELU:
			break;
		case LINEAR:
			break;
		case TANH:
			break;
	}
}


int getWeightIndex(const std::vector<Neuron*>& pointerVec, Neuron* element) {
	for (int i = 0; i < pointerVec.size(); ++i) {
		if (pointerVec[i] == element) {  // Dereference pointer and compare with the element
			return i;
		}
	}
	return -1;  // Return -1 if the element is not found
}


void NeuralNetwork::train(std::vector<std::vector<double>> data, std::vector<std::vector<double>> expected) {

	for (int sampleSize = 0; sampleSize<data.size(); sampleSize++)
	{
		// calculating outputs

		LayerIterator iterator(this->layers);
		// Calculate new outputs
		Layer* layer = iterator.getNext();

		for (int j = 0; j<data[sampleSize].size(); j++){
			layer->neurons[j].output = data[sampleSize][j];
		}

		while (iterator.hasNext()){
			layer = iterator.getNext();
			for (auto n : layer-> neurons){
				double sum = 0.0;
				for (int i = 0; i<n.weights.size(); i++){
					sum += n.inputs[i]->output * n.weights[i];
				}
				n.output = evaluate(sum, n.activation);
			}
		}

		// Calculating gradients

		int s = (int) this->layers.size() - 1;
		while (s>1){
			if (s == this->layers.size() - 1){ // If output layer

				for (int i = 0; i<this->layers[s]->neurons.size(); i++){
					Neuron n = this->layers[s]->neurons[i];
					n.gradient = outputGradient(n.activation, n.output, expected[sampleSize][i]);
				}
			}
			else { // if hidden layers
				for (auto& neuron : this->layers[s]->neurons){
					double out = neuron.output;
					double weightedGrad = 0.0;

					for (auto next : this->layers[s+1]->neurons){
						int idx = getWeightIndex(next.inputs , &neuron);
						double weight = next.weights[idx];
						weightedGrad += next.gradient * weight;
					}
					neuron.gradient = out * (1 - out) * weightedGrad;
				}
			}
			s--;
		}

		// updating weights


		s = (int) this->layers.size() - 1;
		while (s>1) {
			for (auto n: this->layers[s]->neurons){
				for (int i=0; i<n.weights.size(); i++){
					double change = this->learningRate * n.gradient * n.inputs[i]->output;
					n.weights[i] += change;
				}
			}
			s--;
		}

		// updating bias

		s = (int) this->layers.size() - 1;
		while (s>1) {
			for (auto n: this->layers[s]->neurons){
				n.bias += this->learningRate * n.bias;
			}
			s--;
		}


	}




	
	


}



