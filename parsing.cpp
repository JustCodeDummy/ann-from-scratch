
#include "parsing.h"

std::vector<std::string> split(const std::string& str, const std::string& delim){
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


	int sample = 0;
	for (auto & str : lines){
		sample++;
		std::vector<double> row;
		for (const auto& val : split(str, " ")){
			try {
				row.push_back(std::stod(val));
			} catch (std::exception & ex){
				std::cout << sample <<"th sample failed. Remaining: " << lines.size() - sample<< ". Length in previous vs in this sample" << split(lines[sample-2], " ").size() << " : " << split(str, " ").size() << std::endl;

				exit(-1);
			}
		}

		data.push_back(row);
	}

	// Close the file
	file.close();
	return data;
}


void train_test_split(const std::vector<std::vector<double>>& X,
					  const std::vector<std::vector<double>>& y,
					  std::vector<std::vector<double>>& X_train,
					  std::vector<std::vector<double>>& X_test,
					  std::vector<std::vector<double>>& y_train,
					  std::vector<std::vector<double>>& y_test,
					  double test_size = 0.2) {
	// Ensure that X and y have the same size
	if (X.size() != y.size()) {
		std::cout << X.size() << " : " << y.size() << std::endl;
		exit(-1);
	}

	// Create an index vector
	std::vector<int> indices(X.size());
	for (int i = 0; i < X.size(); ++i) {
		indices[i] = i;
	}

	// Shuffle the indices
	std::random_device rd;  // Obtain a random number from hardware
	std::mt19937 g(rd());   // Seed the generator
	std::shuffle(indices.begin(), indices.end(), g);

	// Calculate the size of the test set
	int test_set_size = static_cast<int>(X.size() * test_size);

	// Split the data based on shuffled indices
	for (int i = 0; i < indices.size(); ++i) {
		if (i < X.size() - test_set_size) {
			X_train.push_back(X[indices[i]]);
			y_train.push_back(y[indices[i]]);
		} else {
			X_test.push_back(X[indices[i]]);
			y_test.push_back(y[indices[i]]);
		}
	}

}
