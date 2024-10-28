#include <iostream>
#include "NeuralNetwork.h"
#include "parsing.h"

int main() {
	NeuralNetwork ann;



	std::vector<std::vector<double>> X = loadData("..\\input.txt");
	std::vector<std::vector<double>> y = loadData("..\\output.txt");
	std::cout << X.size() << " : " << y.size() << std::endl;


	std::vector<std::vector<double>> X_train;
	std::vector<std::vector<double>> X_test;
	std::vector<std::vector<double>> y_train;
	std::vector<std::vector<double>> y_test;

	train_test_split(X, y, X_train, X_test, y_train, y_test, 0.3);

	ann.addLayer(new DenseLayer((int) X_train[0].size(), SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(32, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(64, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer(32, SIGMOID, 0.1));
	ann.addLayer(new DenseLayer((int) y_train[0].size(), SOFTMAX, 0.1));


	ann.compile();
	auto start = std::chrono::high_resolution_clock::now();


	Errors result = ann.train(X_train, y_train);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::micro> elapsed = (end - start);

	switch (result){
		case OK:
			std::cout<<"[+] Training finished successfully in " << elapsed.count()  / 1000000  << " seconds" <<std::endl;
			break;
		case NOT_COMPILED:
			std::cerr<<"[-] Model needs to be compiled first"<<std::endl;
			exit(NOT_COMPILED);
		case MISCONFIGURATION:
			std::cerr<<"[-] Misconfiguration, please review your settings" << std::endl;
			exit(MISCONFIGURATION);
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

	start = std::chrono::high_resolution_clock::now();

	ann.predict(X_test, y_test);
	end = std::chrono::high_resolution_clock::now();

	elapsed = (end - start);
	std::cout << "Evaluation took " << elapsed.count() / 1000000 << " seconds." << std::endl;

	return 0;

}



