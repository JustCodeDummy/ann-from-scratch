#pragma once
#include "Neuron.h"
#include "Kernel.h"
#include <cmath>
#include <vector>
#include "Layer.h"

// TODO Add Pooling layer


enum ConvoltionLayerType {
	CONV2D,
	POOLING
};

enum PoolingType {
	NO,
	MAX,
	MIN,
	AVERAGE
};

class ConvolutionLayer {
	public:
		virtual void connectNext(ConvolutionLayer *layer, int kernelSize_) = 0;
		std::vector<std::vector<Neuron>> neurons;
		double learningRate = 0.2;
		virtual ~ConvolutionLayer() = default;
		ActivationFunction function;
		int height, width, padding, kernelSize;
		int stride = 1;
		virtual void updateWeights() = 0;
		std::vector<Kernel> kernels;
		std::vector<std::vector<std::vector<Neuron>>> neurons_;
		virtual std::vector<std::vector<Neuron*>> griddify(std::vector<Neuron*>& inputs, int kSize) = 0;
		ConvoltionLayerType convolutionType;

}
;
class PoolingLayer : public ConvolutionLayer{
	public:
		explicit PoolingLayer(int todo);
		void updateWeights() override;
};

class Conv2dLayer : public ConvolutionLayer {

	private:
		double output;
		std::vector<std::vector<std::vector<double>>> gradients;
		std::vector<std::vector<std::vector<double>>> nextGradients;
		std::vector<Kernel> kernelize(int fIdx);
		PoolingLayer* pulling;
		Conv2dLayer* nextLayer;

	public:
		explicit Conv2dLayer(int height, int width, ActivationFunction function, double bias, int kernelSize, std::function<double(int,int)>&initializer);
		void connectNext(ConvolutionLayer *layer, int kernelSize) override;
		void extractNextGradients(int temp); //Probably can be deleted
		void updateWeights() override;

		std::vector<std::vector<Neuron*>> griddify(std::vector<Neuron*>& inputs, int kSize) override;

		static std::vector<std::vector<double>> extractOutputs(const std::vector<std::vector<Neuron*>> &neurons );

	double convolution(const std::vector<std::vector<double>> &input,
						   int startRow,
						   int startCol,
						   int kernelIndex);

	std::vector<std::vector<std::vector<double>>> featureMaps;
};



class FlattenLayer : public Layer {
	private:
		Conv2dLayer* inputLayer;
		ActivationFunction activation;
		void connectNext(Layer* nextLayer_) override;
		void flatten();
	public:
		~FlattenLayer() override = default;
		FlattenLayer(Conv2dLayer *input);
		std::vector<double> getValues();
		std::vector<Neuron> neurons;


};

