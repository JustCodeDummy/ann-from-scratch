#include "C:\Users\zajcj\.jdks\openjdk-22.0.2\include\jni.h"
#include "C:\Users\zajcj\.jdks\openjdk-22.0.2\include\win32\jni_md.h"
#include <iostream>
#include <vector>
#include "NeuralNetwork.h"
#include "parsing.h"

NeuralNetwork ann;
extern "C" {

// Initialize the neural network
JNIEXPORT void JNICALL Java_NeuralNetworkBridge_initialize(JNIEnv* env, jobject obj) {
	try {

		// Add layers to the neural network
		ann.addLayer(new DenseLayer(32, SIGMOID, 0.1));
		ann.addLayer(new DenseLayer(64, SIGMOID, 0.1));
		ann.addLayer(new DenseLayer(32, SIGMOID, 0.1));
		ann.addLayer(new DenseLayer(10, SOFTMAX, 0.1));
		ann.compile();

	} catch (const std::exception& e) {
		std::cerr << "[ERROR] Exception in initialize: " << e.what() << std::endl;
	}
}

// Load data from file into a 2D Java double array
JNIEXPORT jobject JNICALL Java_NeuralNetworkBridge_loadData(JNIEnv* env, jobject obj, jstring filename) {
	const char* file = env->GetStringUTFChars(filename, nullptr);
	std::vector<std::vector<double>> data = loadData(file);
	env->ReleaseStringUTFChars(filename, file);

	// Convert std::vector to Java 2D double array
	jclass doubleArrayClass = env->FindClass("[D");
	jobjectArray javaDataArray = env->NewObjectArray(data.size(), doubleArrayClass, nullptr);

	for (size_t i = 0; i < data.size(); i++) {
		jdoubleArray row = env->NewDoubleArray(data[i].size());
		env->SetDoubleArrayRegion(row, 0, data[i].size(), &data[i][0]);
		env->SetObjectArrayElement(javaDataArray, i, row);
		env->DeleteLocalRef(row);
	}

	return javaDataArray;
}

// Helper function to convert Java 2D array to C++ vector
std::vector<std::vector<double>> convertJavaArrayToCpp(JNIEnv* env, jobjectArray javaArray) {
	std::vector<std::vector<double>> cppArray;
	jsize outerSize = env->GetArrayLength(javaArray);

	for (jsize i = 0; i < outerSize; i++) {
		jdoubleArray innerArray = (jdoubleArray)env->GetObjectArrayElement(javaArray, i);
		jsize innerSize = env->GetArrayLength(innerArray);
		std::vector<double> row(innerSize);
		env->GetDoubleArrayRegion(innerArray, 0, innerSize, &row[0]);
		cppArray.push_back(row);
		env->DeleteLocalRef(innerArray);
	}

	return cppArray;
}

jobjectArray convertCppArrayToJava(JNIEnv* env, const std::vector<std::vector<double>>& cppArray) {
	jclass doubleArrayClass = env->FindClass("[D");
	jobjectArray javaArray = env->NewObjectArray(cppArray.size(), doubleArrayClass, nullptr);

	for (size_t i = 0; i < cppArray.size(); i++) {
		jdoubleArray row = env->NewDoubleArray(cppArray[i].size());
		env->SetDoubleArrayRegion(row, 0, cppArray[i].size(), &cppArray[i][0]);
		env->SetObjectArrayElement(javaArray, i, row);
		env->DeleteLocalRef(row);
	}

	return javaArray;
}

JNIEXPORT jobjectArray JNICALL Java_NeuralNetworkBridge_trainTestSplit(JNIEnv* env, jobject obj, jobjectArray inputData, jobjectArray outputData, jdouble testSize) {
	// Convert Java arrays to C++ vectors
	std::vector<std::vector<double>> X = convertJavaArrayToCpp(env, inputData);
	std::vector<std::vector<double>> y = convertJavaArrayToCpp(env, outputData);

	// Prepare containers for split data
	std::vector<std::vector<double>> X_train, X_test, y_train, y_test;

	// Call the C++ train_test_split function
	train_test_split(X, y, X_train, X_test, y_train, y_test, testSize);

	// Create a Java array of 4 arrays to hold X_train, X_test, y_train, and y_test
	jclass doubleArrayClass = env->FindClass("[[D");
	jobjectArray resultArray = env->NewObjectArray(4, doubleArrayClass, nullptr);

	// Convert each split C++ vector to Java 2D arrays and add them to the result array
	env->SetObjectArrayElement(resultArray, 0, convertCppArrayToJava(env, X_train)); // X_train
	env->SetObjectArrayElement(resultArray, 1, convertCppArrayToJava(env, X_test));  // X_test
	env->SetObjectArrayElement(resultArray, 2, convertCppArrayToJava(env, y_train)); // y_train
	env->SetObjectArrayElement(resultArray, 3, convertCppArrayToJava(env, y_test));  // y_test

	return resultArray;
}


// JNI function to train the neural network
JNIEXPORT int JNICALL Java_NeuralNetworkBridge_train(JNIEnv* env, jobject obj, jobjectArray inputData, jobjectArray outputData) {

	// Convert Java 2D arrays (inputData and outputData) to C++ vectors
	std::vector<std::vector<double>> X_train = convertJavaArrayToCpp(env, inputData);
	std::vector<std::vector<double>> y_train = convertJavaArrayToCpp(env, outputData);

	// Train the neural network
	Errors result;
	result = ann.train(X_train, y_train);
	return (int32_t)result;
}

JNIEXPORT void JNICALL Java_NeuralNetworkBridge_info(JNIEnv* env, jobject obj) {
	ann.info();
}


// JNI function to predict using the neural network
JNIEXPORT jobject JNICALL Java_NeuralNetworkBridge_predict(JNIEnv* env, jobject obj, jobjectArray inputData) {


	// Convert Java 2D array (inputData) to a C++ vector
	std::vector<std::vector<double>> inputs = convertJavaArrayToCpp(env, inputData);
	std::vector<std::vector<double>> outputs;

	// Predict using the neural network
	try {
		ann.predict(inputs, outputs);
	} catch (const std::exception& e) {
		std::cerr << "[ERROR] Exception during prediction: " << e.what() << std::endl;
		return nullptr;
	}

	// Convert the C++ 2D output vector back to a Java 2D double array
	jclass doubleArrayClass = env->FindClass("[D");
	jobjectArray javaOutputArray = env->NewObjectArray(outputs.size(), doubleArrayClass, nullptr);

	for (size_t i = 0; i < outputs.size(); i++) {
		jdoubleArray row = env->NewDoubleArray(outputs[i].size());
		env->SetDoubleArrayRegion(row, 0, outputs[i].size(), &outputs[i][0]);
		env->SetObjectArrayElement(javaOutputArray, i, row);
		env->DeleteLocalRef(row);
	}

	return javaOutputArray;
}


}
