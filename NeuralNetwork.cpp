#include "NeuralNetwork.h"
#include <cmath>
#include <random>
#include <iostream>


void BaseNeuralNetwork::addLayer(Layer* layer) {
	this->layers.push_back(layer);
}

void BaseNeuralNetwork::info() {
	int c=0;
	for (auto layer : this->layers){
		c++;
		std:: cout << "Properties of layer " << c << std::endl;
		int d=0;
		for (const auto& n : layer->neurons){
			d++;
			std::string w_string;

			w_string += "[";
			for (auto i : n.weights){
				w_string += std::to_string(i);
				w_string += ", ";
			}
			w_string += "]";
			std::vector<int> v;
			std::cout << "\tNeuron (" << d << "): {" << std::endl;
			std::cout << "\t\toutput: " << n.output << std::endl;
			std::cout << "\t\tWeights: " << w_string << std::endl;
			std::cout << "\t\tinputs size: " << n.inputs.size() << std::endl;
			std::cout << "\t\tgradient: " << n.gradient << std::endl;
		}
	}
}

// initialize weights
void BaseNeuralNetwork::compile() {
	if (this->isCompiled){
		return;
	}
	for (int i=1; i<layers.size(); i++){
		this->layers[i -1]->connectNext(this->layers[i]);
	}
	int lid = 1;
	for (auto & layer : this->layers){
		layer->id_ = lid;
		lid++;
		for (auto & n : layer->neurons){
			for (int _ = 0; _<layer->neurons[0].inputs.size(); _++){
				n.weights.push_back(NeuralNetwork::xavier_uniform((long) n.inputs.size(), (long) layer->neurons.size()));
				n.velocities.push_back(0.0);
			}
		}
	}

	this->isCompiled = true;
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

double outputGradient(ActivationFunction condition, double output, double expected, LossFunction loss){

	double o;
	switch (condition) {
		case SIGMOID:
			 o = Sigmoid(output);
			double los;
			los = (expected - o);
			return o * (1 - o) * los;

		case BINARY:
		case RELU:
		case LINEAR:
		case TANH:
			return output - expected;

	}
}

int getWeightIndex(const std::vector<Neuron*>& pointerVec, Neuron* element) {
	for (int i = 0; i < pointerVec.size(); ++i) {
		if (pointerVec[i] == element) {
			return i;
		}
	}
	return -1;
}

Errors BaseNeuralNetwork::train(std::vector<std::vector<double>> data, std::vector<std::vector<double>> expected) {

	 if (!this->isCompiled){
		 return NOT_COMPILED;
	 }
	 if (data.size() != expected.size()){
		 return BAD_DATA;
	 }

	 for (int sample = 0; sample<data.size(); sample++){
		 Errors p = propagate(data[sample]);
		 if (p != OK){
			 return PROPAGATE_FAILED;
		 }
		 Errors bp = backpropagate(expected[sample]);
		 if (bp != OK){
			 return BACK_PROPAGATE_FAILED;
		 }
		 Errors w = updateWeights();
		 if (w != OK){
			 return WEIGHTS_FAILED;
		 }
		 Errors b = updateBias();
		 if (b != OK){
			 return BIAS_FAILED;
		 }
	 }
	 return OK;
}

Errors BaseNeuralNetwork::propagate(std::vector<double> data) {

	try {
		// set input values
		for (int j = 0; j < data.size(); j++) {
			this->layers[0]->neurons[j].output = data[j];
		}
		Layer *layer;
		// calculating outputs
		for (int lay = 1; lay < this->layers.size(); lay++) {
			layer = this->layers[lay];
			int n_count = 0;

			for (auto n: layer->neurons) {
				double sum = 0.0;
				n_count++;
				for (int i = 0; i < n.inputs.size(); i++) {
					double val = n.inputs[i]->output * n.weights[i];;
					sum += val;
				}

				if (n.activation != SOFTMAX) {
					double val = evaluate(sum, n.activation);;
					n.output = val;
				} else {
					n.output = sum;
				}
			}

			if (layer->neurons[0].activation == SOFTMAX) {
				double d = 0;
				for (const auto &n: layer->neurons) {
					d += std::exp(n.output);
				}

				for (auto &n: layer->neurons) {
					n.output = std::exp(n.output) / d;
				}
			}
		}
	} catch (std::exception& ex){
		return PROPAGATE_FAILED;
	}
	return OK;
}



Errors BaseNeuralNetwork::backpropagate(std::vector<double>& expected) {
	try {
		int s = (int) this->layers.size() - 1;
		while (s > 0) {
			if (s == this->layers.size() - 1) { // If output layer
				int i = 0;
				for (auto &n: this->layers[s]->neurons) {
					if (n.activation != SOFTMAX) {
						n.gradient = outputGradient(n.activation, n.output, expected[i],
													CROSS_ENTROPY_MULTICLASS);
						n.pushGradient(n.gradient);
					} else {
						n.gradient = n.output - expected[i];
						n.pushGradient(n.gradient);

					}
					i++;
				}
			} else { // if hidden layers
				for (auto &neuron: this->layers[s]->neurons) {
					double out = neuron.output;
					double weightedGrad = 0.0;

					for (auto &next: this->layers[s + 1]->neurons) {
						int idx = getWeightIndex(next.inputs, &neuron);
						double weight = next.weights[idx];
						weightedGrad += next.gradient * weight;
					}
					neuron.gradient = out * (1 - out) * weightedGrad;
					neuron.pushGradient(neuron.gradient);
				}
			}
			s--;
		}
	} catch (std::exception& ex){
		return BACK_PROPAGATE_FAILED;
	}

	return OK;
}

Errors BaseNeuralNetwork::updateWeights() {
		// updating weights
		int s = (int) this->layers.size() - 1;
		try{
			while (s > 0) {
				for (auto &neuron: this->layers[s]->neurons) {
					for (int i = 0; i < neuron.weights.size(); i++) {
						double gradient = neuron.gradient * neuron.inputs[i]->output + neuron.weights[i] * this->l2;
						neuron.velocities[i] = this->beta * neuron.velocities[i] + (1 - this->beta) * gradient;
						neuron.weights[i] -= this->learningRate * neuron.velocities[i];
					}
				}
				s--;
			}
		} catch(std::exception& ex){
			return WEIGHTS_FAILED;
		}

		return OK;

}

Errors BaseNeuralNetwork::updateBias() {
	int s = (int) this->layers.size() - 1;
	try{
		while (s>0) {
			for (auto &n: this->layers[s]->neurons){
				n.bias += this->learningRate * n.gradient;
			}
			s--;
		}
	}catch (std::exception& ex){
		return BIAS_FAILED;
	}

	return OK;
}

int BaseNeuralNetwork::predict(const std::vector<double>& sample) {

	propagate(sample);

	int index = -1;
	double max = -1000;
	std::vector<Neuron> outputs = layers[ layers.size()-1]->neurons;
	for (int i=0; i<outputs.size(); i++){
		if (outputs[i].output > max){
			index = i;
			max = outputs[i].output;
		}
	}
	return index;
}

void BaseNeuralNetwork::predict(const std::vector<std::vector<double>> &inputs,
							const std::vector<std::vector<double>> &outputs) {
	int correctly = 0;
	for (int i=0; i<inputs.size(); i++){
		if (correct(outputs[i]) == predict(inputs[i])){
			correctly += 1;
		}
	}

	std::cout << "Neural network classified " << correctly << " samples correctly, which is " << (correctly /inputs.size()) << "%" << std::endl;

}

int BaseNeuralNetwork::correct(const std::vector<double> &values) {
		for (int i=0; i<values.size(); i++){
			if (values[i] > 0.0){
				return i;
			}
		}
		return -1;

}

BaseNeuralNetwork::BaseNeuralNetwork(std::vector<Layer*>& convLayers) {
	this->isCompiled = true;
	this->layers = convLayers;

}

Errors BaseNeuralNetwork::propagateConvolution() {

	std::vector<double> data(this->layers[0]->neurons.size(), 0);
	for (int i = 0; i<data.size(); i++){
		data[i] = this->layers[0]->neurons[i].output;
	}
	Errors statusCode = this->propagate(data);

	return statusCode;

}
