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
		int id_=0;
 };

class DenseLayer : public Layer {
	public:
		explicit DenseLayer(int units, ActivationFunction function, double bias);
};


class ConvolutionLayer {
	public:
		std::vector<std::vector<Neuron>> neurons;
		void connectNext(ConvolutionLayer *layer, int kernelSize_, int stride);
		virtual ~ConvolutionLayer() = default;
		ActivationFunction function;
		int height, width;
		int kernelSize;
		int padding;


};


class Conv2dLayer : public ConvolutionLayer {

	private:

		double kernel[9][9];
	public:
		explicit Conv2dLayer(int height, int width, ActivationFunction function, double bias, int kernelSize);


};

