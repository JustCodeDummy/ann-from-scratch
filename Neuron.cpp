#include "Neuron.h"

#include <utility>


Neuron::Neuron(std::function<double(double)> activation, double bias, NeuronType type) {
	this->bias = bias;
	this->activation = std::move(activation);
	this->type = type;

}


ArrayList<Layer> layers = new ArrayList<>();
layers.push(new denseLayer(some_params))
layers.push(new denseLayer(some_params))
layers.push(new denseLayer(some_params))

while (layers.has_next){
	layer.connectNext(layers.getNext());
}
