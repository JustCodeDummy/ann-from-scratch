
#include "parsing.h"
#include "opencv2/opencv.hpp"

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

	if (!file.is_open()) {
		std::cerr << "Unable to open the file!" << std::endl;
		std::cout << "Current path: " << std::filesystem::current_path() << std::endl;
		exit(-1);
	}

	std::vector<std::string> lines;
	std::string line;

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
	if (X.size() != y.size()) {
		std::cout << X.size() << " : " << y.size() << std::endl;
		exit(-1);
	}

	std::vector<int> indices(X.size());
	for (int i = 0; i < X.size(); ++i) {
		indices[i] = i;
	}

	std::random_device rd;  // Obtain a random number from hardware
	std::mt19937 g(rd());   // Seed the generator
	std::shuffle(indices.begin(), indices.end(), g);

	int test_set_size = static_cast<int>((int) X.size() * test_size);

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

void transformConv2dGrayScale(int height, int width, std::string& filepath, std::vector<std::vector<double>>& data){
	cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);

	if (image.empty()){
		std::cerr << "[-] Unable to load this image" << std::endl;
		exit(-9);
	}

	cv::resize(image, image, cv::Size(width, height));

	for (int i = 0; i<width; i++){
		for (int j = 0; j<height; j++){
			data[i][j] = static_cast<double>(image.at<uchar>(i,j)) / 255.00;
		}
	}






}
