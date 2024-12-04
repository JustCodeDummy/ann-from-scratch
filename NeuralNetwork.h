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
	MISCONFIGURATION = -2,
	BIAS_FAILED = -3,
	WEIGHTS_FAILED = -4,
	BACK_PROPAGATE_FAILED = -5,
	PROPAGATE_FAILED = -6,
	BAD_DATA = -7
};

enum LossFunction{
	MSE,
	CROSS_ENTROPY_MULTICLASS,
	CROSS_ENTROPY_BINARY
};

class NeuralNetwork{
	public:
		double beta = 0.9;
		bool isCompiled = false;
		virtual ~NeuralNetwork() = default;
		double l2 = 0.01;
		// TODO Dynamic learning rate, DROPOUT regularization
		double learningRate = 0.1;
		static double xavier_uniform(long n_in, long n_out);
		virtual void compile() = 0;
};


class BaseNeuralNetwork: public NeuralNetwork{
	private:
		std::vector<Layer*> layers;
		int correct(const std::vector<double> &sample);
		Errors updateWeights();
		Errors updateBias();
		Errors propagate(std::vector<double> data);



	public:
		BaseNeuralNetwork() = default;

		explicit BaseNeuralNetwork(std::vector<Layer*>& layers);
		Errors propagateConvolution();
		Errors backpropagate(std::vector<double>& expected);


		int predict(const std::vector<double>& sample);

		void predict(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs);

		void addLayer(Layer* layer);

		void info();

		void compile() override;

		Errors train(std::vector<std::vector<double>> data, std::vector<std::vector<double>> expected);

};
