#pragma once
#include "Neuron.h"
#include "Layer.h"
#include <vector>

// TODO implement different algorithms and add them to this enum
enum LearningAlgorithm {
	BACK_PROPAGATION,
};

enum Errors{
	OK=0,
	NOT_COMPILED = -1,
	MISCONFIGURATION = -2
};

enum LossFunction{
	MSE,
	CROSS_ENTROPY_MULTICLASS,
	CROSS_ENTROPY_BINARY
};

class NeuralNetwork{
	private:
		std::vector<Layer*> layers;
		static double xavier_uniform(long n_in, long n_out);
		bool isCompiled = false;
		class LayerIterator {
			public:
				explicit LayerIterator(std::vector<Layer*>& layers) : layers_(layers), index_(0) {}

				[[nodiscard]] bool hasNext() const {
					return index_ < layers_.size() - 1;  // Stops before the last layer
				}

				Layer* getNext() {
					return layers_[index_++];  // Return the current layer and increment the index
				}

			private:
				std::vector<Layer*>& layers_;
				std::size_t index_;
			};


	public:
		NeuralNetwork() = default;

		double l2 = 0.01;
 		// TODO Dynamic learning rate, DROPOUT regularization
		double learningRate = 0.1;

		void addLayer(Layer* layer);

		void info();

		void compile();

		Errors train(std::vector<std::vector<double>> data, std::vector<std::vector<double>> expected);

};