#pragma once

#include "Neuron.h"
#include <vector>
#include <functional>
// Parent "abstract" class

enum PaddingType {
	NO_PADDING,
	EVEN,
	DILATED
};

 class Layer {
	public:
		std::vector<Neuron> neurons;
		virtual void connectNext(Layer* nextLayer) = 0;
		virtual ~Layer() = default;

 };


class InputLayer : public Layer {
	public:
		explicit InputLayer(int units, const std::function<double(double)>& activation, double bias);
};



class DenseLayer : public Layer {

	public:
		explicit DenseLayer(int units, const std::function<double(double)>& activation, double bias);
		void connectNext(Layer* nextLayer) override;

};


class OutputLayer : public Layer {
	public:
		explicit OutputLayer(int units, const std::function<double(double)> &activation, double bias);

};





// TODO convolutional1d/2d/lstm,recurrent,...