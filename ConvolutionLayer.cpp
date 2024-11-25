#include "ConvolutionLayer.h"


Conv2dLayer::Conv2dLayer(int height, int width, ActivationFunction function, double bias, int kernelSize, std::function<double(int,int)>&initializer){
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
	this->kernels = std::vector<Kernel>(kernelSize, Kernel((size_t) kernelSize, (size_t) kernelSize, initializer));


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
}

double Conv2dLayer::convolution(const std::vector<std::vector<double>>& input,
								int startRow,
								int startCol,
								int kernelIndex) {
	auto& kernel =  this->kernels[kernelIndex];
	size_t k_rows = kernel.size();
	size_t k_cols = kernel[0].size();
	double summation = 0.0;
	for (size_t row = 0; row < k_rows; row++) {
		for (size_t col = 0; col < k_cols; col++) {
			summation += input[startRow + row][startCol + col] * kernel[row][col];
		}
	}
	return summation;
}

void Conv2dLayer::propagate() {

	int p_rows = (int) this->neurons.size();
	int p_cols = (int) this->neurons[0].size();

	int outRows = (p_rows - this->kernelSize) / stride + 1;
	int outCols = (p_cols - this->kernelSize) / stride + 1;
	for (int idx = 0; idx<this->kernels.size(); idx++){
		std::vector<std::vector<double>> convolved(outRows, std::vector<double>(outCols, 0.0));
		for (int row=0; row < outRows; row++){
			for (int col=0; col < outCols; col++){
				int s_row = row * this->stride;
				int s_col = col * this->stride;
				auto doubles = extractOutputs(griddify(this->neurons_[idx][s_row][s_col].inputs, (int) kernels[idx].size()));
				convolved[row][col] = convolution(doubles, s_row, s_col, idx);
			}
		}
		this->featureMaps[idx] = convolved;
	}

	for (int map = 0; map < kernels.size(); map++){
		for (int row = 0; row < outRows; row++) {
			for (int col = 0; col < outCols; col++) {
				int neuronRow = row * this->stride + this->padding;
				int neuronCol = col * this->stride + this->padding;
				if (neuronRow < this->neurons.size() && neuronCol < this->neurons[0].size()) {
					this->neurons_[map][neuronRow][neuronCol].output = featureMaps[map][row][col];
				}
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
std::vector<std::vector<Neuron *>> Conv2dLayer::griddify(std::vector<Neuron*> &inputs, int kSize) {

	int n_row = (int) this->kernels[kSize].size();
	int n_col = (int) this->kernels[kSize][0].size();

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


std::vector<Kernel> Conv2dLayer::kernelize(int fIdx) {

	int n_row = (int) this->neurons_[fIdx].size();
	int n_col = (int) this->neurons_[fIdx][0].size();
	auto featureMap = featureMaps[fIdx];

	int feat_row = (int) featureMap.size();
	int feat_col = (int) featureMap[0].size();

	int kSize = (int) kernels[fIdx].size();

	std::vector<std::vector<double>> grid(n_row, std::vector<double>(n_col, 0));

	std::vector<Kernel> kernels_;

	kernels_.resize((size_t) std::pow(featureMaps[fIdx].size(), 2));

	for (int i = 0; i<n_row; i++){
		for (int j=0; j<n_col; j++){
			grid[i][j] = neurons_[fIdx][i][j].output;
		}
	}

	for (size_t k_row = 0; k_row<feat_row; k_row++){
		for (size_t k_col = 0; k_col < feat_col;k_col++){
			std::vector<std::vector<double>> doubles(kSize, std::vector<double>(kSize, 0));
			for (size_t i = 0; i<kSize; i++){
				for (size_t j = 0; j<kSize; j++){
					doubles[i][j] = grid[k_row + i][k_col + j];
				}
			}
			kernels_.emplace_back(doubles);
		}
	}
	return kernels_;
}

void Conv2dLayer::backpropagate() {
	for (int kIdx = 0; kIdx < this->kernels.size(); kIdx++){
		auto kernels_ = kernelize(kIdx);
		auto s = (size_t) std::sqrt((size_t)kernels_.size());
		Kernel gradients(kernelSize);
		for (int row = 0; row < s; row++){
			for (size_t col = 0; col < s; col++){
				Kernel t = kernels_[row * kernelSize + col] * featureMaps[kIdx][row][col];
				gradients += t;
			}
		}
		Kernel k= gradients * this->learningRate;
		this->kernels[kIdx] -= k;
	}
}
