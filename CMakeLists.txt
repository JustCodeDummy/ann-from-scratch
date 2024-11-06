cmake_minimum_required(VERSION 3.26)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 23)

include_directories("C:/Program Files/Java/jdk-21/include" "C:/Program Files/Java/jdk-21/include/win32")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(NeuralNetwork main.cpp
        Neuron.cpp
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

target_link_libraries(NeuralNetwork ${OpenCV_LIBS})
