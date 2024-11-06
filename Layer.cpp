//
// Created by fsociety on 03/09/2024.
//

#include "Layer.h"
#include "parsing.h"
DenseLayer::DenseLayer(int units, ActivationFunction activation, double bias) {
	for (int i = 0; i<units; i++){
		this->neurons.emplace_back(activation, bias, i+1);
	}
}

void Layer::connectNext(Layer *nextLayer) {
	for (auto &input: this->neurons){
		for (auto &neuron : nextLayer->neurons) {
			neuron.inputs.push_back(&input);
		}
	}
}


Conv2dLayer::Conv2dLayer(int height, int width, ActivationFunction function, double bias, int kernelSize) {
	this->height = height;
	this->width = width;

	// Validate kernel size
	if (kernelSize > 9) {
		std::cout << "[!] Kernel size too large, using 9 instead" << std::endl;
		kernelSize = 9;
	}

	if (kernelSize < 3) {
		std::cout << "[-] Kernel size too small, using 3 instead" << std::endl;
		kernelSize = 3;
	}

	this->kernelSize = kernelSize;
	this->padding = (kernelSize - 1) / 2;

	int paddedHeight = height + 2 * padding;
	int paddedWidth = width + 2 * padding;

	this->neurons.resize(paddedWidth); // Resize outer vector to padded width
	for (int i = 0; i < paddedWidth; ++i) {
		this->neurons[i].resize(paddedHeight); // Resize each inner vector to padded height
		for (int j = 0; j < paddedHeight; ++j) {
			if (i >= padding && i < paddedWidth - padding && j >= padding && j < paddedHeight - padding) {
				this->neurons[i][j] = Neuron(function, bias, i + 1);
			}
		}
	}
}

void ConvolutionLayer::connectNext(ConvolutionLayer *layer, int kernelSize_, int stride) {
	int outputWidth = layer->width;
	int outputHeight = layer->height;

	for (int col_n = 0; col_n < outputWidth; col_n++) {
		for (int row_n = 0; row_n < outputHeight; row_n++) {
			int startCol = col_n * stride;
			int startRow = row_n * stride;

			for (int col = 0; col < kernelSize_; col++) {
				for (int row = 0; row < kernelSize_; row++) {
					int inputCol = startCol + col;
					int inputRow = startRow + row;
					if (inputCol >= 0 && inputCol < this->width && inputRow >= 0 && inputRow < this->height) {
						layer->neurons[col_n][row_n].inputs.push_back(&this->neurons[inputCol][inputRow]);
					}
				}
			}
		}
	}
}


