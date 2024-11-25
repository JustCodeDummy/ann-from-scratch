#pragma once
#include "Neuron.h"
#include "Kernel.h"
#include <cmath>

class ConvolutionLayer {
public:
	virtual void connectNext(ConvolutionLayer *layer, int kernelSize_) = 0;
	std::vector<std::vector<Neuron>> neurons;
	double learningRate = 0.2;
	virtual ~ConvolutionLayer() = default;
	ActivationFunction function;
	int height, width;
	int kernelSize;
	int padding;
	int stride = 1;
	virtual void propagate() = 0;
	virtual void backpropagate() = 0;
};

class Conv2dLayer : public ConvolutionLayer {

	private:
		std::vector<std::vector<std::vector<double>>> featureMaps;
		std::vector<Kernel> kernels;
		std::vector<std::vector<double>> output;

		std::vector<std::vector<std::vector<Neuron>>> neurons_;
		static std::vector<std::vector<double>> extractOutputs(const std::vector<std::vector<Neuron*>> &neurons );
		std::vector<std::vector<Neuron*>> griddify(std::vector<Neuron*>& inputs, int kSize);
		double convolution(const std::vector<std::vector<double>> &input,
						   int startRow,
						   int startCol,
						   int kernelIndex);
	public:
		explicit Conv2dLayer(int height, int width, ActivationFunction function, double bias, int kernelSize, std::function<double(int,int)>&initializer);
		void propagate() override;
		void connectNext(ConvolutionLayer *layer, int kernelSize) override;
		void backpropagate() override;
		std::vector<Kernel> kernelize(int fIdx);
};