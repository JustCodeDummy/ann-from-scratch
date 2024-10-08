#include <iostream>
#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
using namespace ActivationFunctions;

int main() {
	NeuralNetwork ann;

	std::vector<std::vector<double>> data;
	std::vector<double> d1;
	d1.push_back(0.3);
	d1.push_back(0.2);

	data.push_back(d1);

	std::vector<double> output;
	output.push_back(0);
	output.push_back(1);
	output.push_back(0);

	std::vector<std::vector<double>> output_data;
	output_data.push_back(output);

	ann.addLayer(new DenseLayer(2, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(3, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(3, SOFTMAX, 0.1));


	ann.compile();

	Errors result = ann.train(data, output_data);

	switch (result){

		case OK:
			std::cout<<"[+] Yeet yeet"<<std::endl;
			break;
		case NOT_COMPILED:
			std::cout<<"[-] Compile fucker first!"<<std::endl;
			break;
		case MISCONFIGURATION:
			std::cout<<"[!] Why dfq was that thrown????" << std::endl;
			break;
	}


	return 0;

}


