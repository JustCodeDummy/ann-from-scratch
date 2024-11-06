#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<std::string> split(const std::string& str, const std::string& delim);

std::vector<std::vector<double>> loadData(const std::string& filename);

void train_test_split(const std::vector<std::vector<double>>& X,
					  const std::vector<std::vector<double>>& y,
					  std::vector<std::vector<double>>& X_train,
					  std::vector<std::vector<double>>& X_test,
					  std::vector<std::vector<double>>& y_train,
					  std::vector<std::vector<double>>& y_test,
					  double test_size);

void transformConv2dGrayScale(int height, int weight, std::string &filepath, double ptr[]);


