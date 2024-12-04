#include "ConvolutionLayer.h"
#include <cmath>





class ConvolutionalNeuralNetwork {
	private:
		double learningRate = 0.1;
		bool isCompiled = false;
		FlattenLayer flattenLayer;
		std::vector<Conv2dLayer*> convolutionalLayers;
		std::vector<Layer*> denseLayers;

	public:
		explicit ConvolutionalNeuralNetwork();
		void updateFilters();
		void updateBias();
		void propagate();
		void backpropatgation();
		void compile();
		void addLayer(Conv2dLayer& layer){
			this->convolutionalLayers.push_back(&(layer));
		}
		void addLayer(DenseLayer& layer){
			this->denseLayers.push_back(&(layer));
		}
};
