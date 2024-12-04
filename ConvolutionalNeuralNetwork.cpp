//
// Created by fsociety on 01/12/2024.
//

#include "ConvolutionalNeuralNetwork.h"
#include "NeuralNetwork.h"

double evaluate_(const double x, ActivationFunction condition){
	switch (condition) {
		case SIGMOID:
			return 0; // TODO actual sigmoid
		case BINARY:
			return x > 0 ? 1 : 0;
		case RELU:
			return x > 0 ? x : 0;
		case LINEAR:
			return x;
		case TANH:
			return std::tanh(x);
	}
}


void ConvolutionalNeuralNetwork::propagate() { // TODO change return type to Errors for handling, like in NeuralNetwork.h
	try {
		for (auto & layer : this->convolutionalLayers) {
			int p_stride = layer->stride;
			int p_padding = layer->padding;
			int p_rows = (int) layer->neurons.size();
			int p_cols = (int) layer->neurons[0].size();
			int outRows = (p_rows - layer->kernelSize) / p_stride + 1;
			int outCols = (p_cols - layer->kernelSize) / p_stride + 1;
			for (int idx = 0; idx < layer->kernels.size(); idx++) {
				std::vector<std::vector<double>> convolved(outRows, std::vector<double>(outCols, 0.0));
				for (int row = 0; row < outRows; row++) {
					for (int col = 0; col < outCols; col++) {
						int s_row = row * p_stride;
						int s_col = col * p_stride;
						auto doubles = Conv2dLayer::extractOutputs(
								layer->griddify(layer->neurons_[idx][s_row][s_col].inputs,
												(int) layer->kernels[idx].size()));

						convolved[row][col] = layer->convolution(doubles, s_row, s_col, idx);
					}
				}
				layer->featureMaps[idx] = convolved;
			}

			for (int map = 0; map < layer->kernels.size(); map++) {
				for (int row = 0; row < outRows; row++) {
					for (int col = 0; col < outCols; col++) {
						int neuronRow = row * p_stride + p_padding;
						int neuronCol = col * p_stride + p_padding;
						if (neuronRow < layer->neurons_.size() && neuronCol < layer->neurons_[0].size()) {
							layer->neurons_[map][neuronRow][neuronCol].output = layer->featureMaps[map][row][col];
						}
					}
				}
			}
		}
		BaseNeuralNetwork ann(this->denseLayers);
		// TODO think through compile function
		ann.propagateConvolution();


	} catch (std::exception& ex){
		return;// PROPAGATE_FAILED;
	}
	//return OK;
}




void ConvolutionalNeuralNetwork::compile() {

	if (isCompiled){
		return;
	}

	for (int i = 1; i<this->convolutionalLayers.size(); i++ ){
		this->convolutionalLayers[i - 1]->connectNext(this->convolutionalLayers[i], this->convolutionalLayers[i]->kernelSize);
	}

	flattenLayer = FlattenLayer(this->convolutionalLayers[convolutionalLayers.size() - 1]);

	denseLayers.insert(denseLayers.begin(), &(flattenLayer));

	for (int i = 1; i<this->denseLayers.size(); i++){
		denseLayers[i-1]->connectNext(denseLayers[i]);
	}

	int lid = 1;
	for (auto &layer : this->denseLayers){
		layer->id_ = lid;
		for (auto &neuron : layer->neurons){
			for (int _ = 0; _<layer->neurons[0].inputs.size(); _++){
				neuron.weights.push_back(NeuralNetwork::xavier_uniform((long) neuron.inputs.size(), (long) layer->neurons.size()));
				neuron.velocities.push_back(0.0);
			}
		}
	}

	for (auto& neuron : flattenLayer.neurons){
		neuron.weights = std::vector<double>(flattenLayer.neurons.size(), 1.00);
	}

}


