#pragma once

#include "Neuron.h"
#include <vector>
#include <functional>



class Layer {
	public:
		std::vector<Neuron> neurons;
		virtual void connectNext(Layer* nextLayer) = 0;
		virtual ~Layer() = default;
		ActivationFunction function;
		int id_=0;
 };

class DenseLayer : public Layer {
	public:
		explicit DenseLayer(int units, ActivationFunction function, double bias);
};



