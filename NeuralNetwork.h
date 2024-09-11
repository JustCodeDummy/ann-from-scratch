#pragma once
#include "Neuron.h"
#include "Layer.h"
#include <vector>

// TODO implement different algorithms and add them to this enum
enum LearningAlgorithm {
	BACK_PROPAGATION,
};


class NeuralNetwork{
	private:
		std::vector<Layer*> layers;
		static double xavier_uniform(long n_in, long n_out);

		class LayerIterator {
			public:
				LayerIterator(std::vector<Layer*>& layers) : layers_(layers), index_(0) {}

				bool hasNext() const {
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

		void addLayer(Layer* layer);

		void info();

		void compile();

		void set();

		void train(LearningAlgorithm algorithm);

};