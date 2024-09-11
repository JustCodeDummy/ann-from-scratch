#include <iostream>
#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
using namespace ActivationFunctions;

int main() {
	NeuralNetwork ann;

	ann.addLayer(new InputLayer(4, Sigmoid, 0.0));
	ann.addLayer(new DenseLayer(4, ReLU, 0.01));
	ann.addLayer(new DenseLayer(4, ReLU, 0.01));
	ann.addLayer(new DenseLayer(1, Sigmoid, 0.0));

	ann.compile();
	return 0;

}


