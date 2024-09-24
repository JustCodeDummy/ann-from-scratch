#pragma once

#include "Neuron.h"
#include <vector>
#include <functional>




class Layer {
	public:
		std::vector<Neuron> neurons;
		void connectNext(Layer* nextLayer);
		virtual ~Layer() = default;
		ActivationFunction function;
 };


class DenseLayer : public Layer {
	public:
		explicit DenseLayer(int units, ActivationFunction function, double bias);
};

