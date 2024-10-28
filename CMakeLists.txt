cmake_minimum_required(VERSION 3.26)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 23)

add_executable(NeuralNetwork main.cpp
        Neuron.cpp
        Helpers.cpp
        Helpers.h
        ActivationFunctions.h
        ActivationFunctions.cpp
        Layer.cpp
        Layer.h
        NeuralNetwork.cpp
        NeuralNetwork.h
        LearningAlgorithms.cpp
        LearningAlgorithms.h
        LearningAlgorithms.h
        parsing.cpp
        parsing.h
        NeuralNetworkBridge.cpp
        NeuralNetworkBridge.cpp
)

include_directories("C:/Program Files/Java/jdk-21/include" "C:/Program Files/Java/jdk-21/include/win32")

