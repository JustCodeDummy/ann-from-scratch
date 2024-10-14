#include <iostream>
#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include <fstream>
#include <filesystem>
using namespace ActivationFunctions;

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
	std::vector<std::string> result;
	size_t start = 0;

	for (size_t found = str.find(delim); found != std::string::npos; found = str.find(delim, start))
	{
		result.emplace_back(str.begin() + start, str.begin() + found);
		start = found + delim.size();
	}
	if (start != str.size())
		result.emplace_back(str.begin() + start, str.end());
	return result;
}

std::vector<std::vector<double>> loadData(const std::string& filename){
	std::ifstream file(filename);

	// Check if the file was opened successfully
	if (!file.is_open()) {
		std::cerr << "Unable to open the file!" << std::endl;
		std::cout << "Current path: " << std::filesystem::current_path() << std::endl;
		exit(-1);
	}

	// Vector to store lines
	std::vector<std::string> lines;
	std::string line;

	// Read the file line by line
	while (std::getline(file, line)) {
		lines.push_back(line);
	}

	std::vector<std::vector<double>> data;
	data.reserve(lines.size());



	for (auto & str : lines){
		std::vector<double> row;
		for (const auto& val : split(str, " ")){
			row.push_back(std::stod(val));
		}
		data.push_back(row);
	}

	// Close the file
	file.close();
	return data;
}

int main() {
	NeuralNetwork ann;


	std::vector<std::vector<double>> input = loadData("..\\inputs.txt");
	std::vector<std::vector<double>> output = loadData("..\\outputs.txt");


	ann.addLayer(new DenseLayer(6, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(8, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(8, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(4, SOFTMAX, 0.1));


	ann.compile();

	Errors result = ann.train(input, output);

	switch (result){
		case OK:
			std::cout<<"[+] Training finished successfully"<<std::endl;
			break;
		case NOT_COMPILED:
			std::cerr<<"[-] Model needs to be compiled first"<<std::endl;
			break;
		case MISCONFIGURATION:
			std::cerr<<"[-] Misconfiguration, please review your settings" << std::endl;
			break;
		case BIAS_FAILED:
			std::cerr<<"[!] Updating bias failed, exiting...";
			exit(BIAS_FAILED);
		case WEIGHTS_FAILED:
			std::cerr<<"[!] Updating weights failed, exiting...";
			exit(WEIGHTS_FAILED);
		case BACK_PROPAGATE_FAILED:
			std::cerr << "[!] Backpropagation failed, exiting...";
			exit(BACK_PROPAGATE_FAILED);
		case PROPAGATE_FAILED:
			std::cerr << "[!] Propagation failed, exiting...";
			exit(PROPAGATE_FAILED);
		case BAD_DATA:
			std::cerr <<"[-] Input and output sizes do not match, exiting...";
			exit(BAD_DATA);
	}
	return 0;

}


