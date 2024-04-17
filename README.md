# ml-linalg
## Introduction
The code in this repo implements the back propagation training algorithm for a Neural Network using the [std::linalg library](https://github.com/kokkos/stdBLAS) reference implementation. The repo includes test code and an example of how to use the library. This code is primarly intended for eductionaly purposes rather than real world projects. 

This code is an evolution of [Machine-Learning-CPP code](https://github.com/GarethRichards/Machine-Learning-CPP).

## Requirements

  - CMake >= 3.17 (earlier versions may work, but are not tested)
  - C++ build environment that supports C++17 or greater

## Tested compilers

   - GCC14.
   - MSVC 2022

## Brief build instructions

   1. Follow the build instructions on the [stdBLAS page](https://github.com/kokkos/stdBLAS).

## Using this library
This is a header only libray to use it add the _include_ directory of this repo to your projects list of includes. (To do work out how to do this via *cmake* _FetchContent_)

See the code in [example/NeuralNet3.cpp](example/NeuralNet3.cpp) for an example on how to use the `NeuralNet::Network` template.

A short article describing the differences between [std::linalg library](https://github.com/kokkos/stdBLAS) and [boost:ublas](https://www.boost.org/doc/libs/1_84_0/libs/numeric/ublas/doc/) versions of this code can be found [here](LINALG_BOOST_UBLAS.md).

## Basic usage

``` cpp
#include "NeuralNet.h"

// define the network type
using NeuralNet1 = NeuralNet::Network<float, NeuralNet::QuadraticCost<float, NeuralNet::ReLUActivation<float>>, NeuralNet::ReLUActivation<float>>;
using NetData = std::vector<NeuralNet::Network_interface<float>::TrainingData>;
using NetDataMem = std::vector<std::pair<std::vector<float>, std::vector<float>>>;

NetData td;
NetData testData;
NetDataMem training_data_mem;
NetDataMem test_data_mem;

// 
// Load the training data
//
// define the network
NeuralNet1 net({ 784, 100, 46, 21, 10 });
// set up the training parameters
float Lmbda = 0.1f; 
float eta = 0.03f; 
int iterations = 20;
int mini_batch_size = 100;

net.SGD(td.begin(), td.end(), iterations, mini_batch_size, eta, Lmbda,
            [&periodStart, &Lmbda, &testData, &td](const NeuralNet1 &network, int Epoch, float &currenctEta) {
         // feedback Lambda can be 
         std::cout << "Epoch " << Epoch << "\n";
         std::cout << "Training accuracy : " << network.accuracy(td.begin(), td.end()) << " / " << td.size() << "\n";
    });

// network is now trained and ready to be used
```