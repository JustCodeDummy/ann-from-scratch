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
	this->width = width;
	this->height = height;

	if (kernelSize % 2 == 0){
		kernelSize--;
	}

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

	this->neurons.resize(paddedWidth);
	for (int i = 0; i < paddedWidth; ++i) {
		this->neurons[i].resize(paddedHeight);
		for (int j = 0; j < paddedHeight; ++j) {
			if (i >= padding && i < paddedWidth - padding && j >= padding && j < paddedHeight - padding) {
				this->neurons[i][j] = Neuron(function, bias, i + 1);
			} else {
				this->neurons[i][j] = Neuron();
			}
		}
	}
	this->kernel = std::vector<std::vector<double>>(kernelSize, std::vector<double>(kernelSize, 0.0));
	for (int i = 0; i <this->kernelSize; i++){
		for (int j= 0; j < kernelSize; j++){
			this->kernel[i][j] = (std::sqrt(6.0 * kernelSize / (int) (this->neurons.size() * this->neurons[0].size())));
		}
	}

}

double Conv2dLayer::convolution(const std::vector<std::vector<double>>& input,
								int startRow,
								int startCol) {
	double summation = 0.0;

	// Loop through the kernel
	for (int i = 0; i < kernel.size(); i++) {
		for (int j = 0; j < kernel[0].size(); j++) {
			// Apply kernel to the corresponding input region
			summation += input[startRow + i][startCol + j] * kernel[i][j];
		}
	}

	return summation;
}

void Conv2dLayer::update() {

	int p_rows = this->neurons.size();
	int p_cols = this->neurons[0].size();

	int outRows = (p_rows - this->kernelSize) / stride + 1;
	int outCols = (p_cols - this->kernelSize) / stride + 1;
	std::vector<std::vector<double>> convolved(outRows, std::vector<double>(outCols, 0.0));
	for (int i=0; i < outRows; i++){
		for (int j=0; j < outCols; j++){
			int s_row = i*this->stride;
			int s_col = j*this->stride;
			auto doubles = extractOutputs(griddify(this->neurons[s_row][s_col].inputs));
			convolved[i][j] = convolution(doubles, s_row, s_col);
		}
	}
	for (int i = 0; i < outRows; i++) {
		for (int j = 0; j < outCols; j++) {
			int neuronRow = i * this->stride + this->padding;
			int neuronCol = j * this->stride + this->padding;
			if (neuronRow < this->neurons.size() && neuronCol < this->neurons[0].size()) {
				this->neurons[neuronRow][neuronCol].output = convolved[i][j];
			}
		}
	}
}

std::vector<std::vector<double>> Conv2dLayer::extractOutputs(const std::vector<std::vector<Neuron*>> &inputs) {
	int rows = (int) inputs.size();
	int cols = (int) inputs[0].size();
	std::vector<std::vector<double>> outputs(rows, std::vector<double>(cols, 0.0));
	for (int row = 0; row<rows; row++){
		for (int col = 0; col<cols; col++){
			outputs[row][col] = inputs[row][col]->output;
		}
	}
	return outputs;
}

std::vector<std::vector<Neuron *>> Conv2dLayer::griddify(std::vector<Neuron*> &inputs) {

	int n_row = this->kernel.size();
	int n_col = this->kernel[0].size();

	std::vector<std::vector<Neuron *>> grid(n_row, std::vector<Neuron*>(n_col, nullptr));

	for (int i = 0; i<n_row; i++){
		for (int j=0; j<n_col; j++){
			grid[i][j] = inputs[i*n_row + j];
		}
	}
	return grid;
}

void Conv2dLayer::connectNext(ConvolutionLayer *layer, int kernelSize) {
	int outputWidth = layer->width;
	int outputHeight = layer->height;

	for (int col_n = 0; col_n < outputWidth; col_n++) {
		for (int row_n = 0; row_n < outputHeight; row_n++) {
			int startCol = col_n * this->stride;
			int startRow = row_n * stride;

			for (int col = 0; col < kernelSize; col++) {
				for (int row = 0; row < kernelSize; row++) {
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




