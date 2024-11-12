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
		virtual void connectNext(ConvolutionLayer *layer, int kernelSize_) = 0;
		virtual ~ConvolutionLayer() = default;
		ActivationFunction function;
		int height, width;
		int kernelSize;
		int padding;
		int stride = 1;
		virtual void update() = 0;

};

class Conv2dLayer : public ConvolutionLayer {

	private:
		std::vector<std::vector<double>> kernel;

		std::vector<std::vector<double>> output;
		static std::vector<std::vector<double>> extractOutputs(const std::vector<std::vector<Neuron*>> &neurons );
		std::vector<std::vector<Neuron*>> griddify(std::vector<Neuron*>& inputs);
		double convolution(const std::vector<std::vector<double>> &input,
							int startRow,
							int startCol);
	public:
		explicit Conv2dLayer(int height, int width, ActivationFunction function, double bias, int kernelSize);
		void update() override;
		void connectNext(ConvolutionLayer *layer, int kernelSize) override;


};

