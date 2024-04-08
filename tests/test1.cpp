//
// Code is written to be a C++ version of network2.py from
// http://neuralnetworksanddeeplearning.com/chap3.html Variable and functions
// names follow the names used in the original Python
//
// This implementation aims to be slight better C++ rather than Python code
// ported to C++
//
// Uses the experimental/linalg extension for linear algebra operations

#include "NeuralNet.h"
#include "mnist_loader.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace NeuralNet;
using namespace std::filesystem;

using NeuralNet1 = NeuralNet::Network<float, NeuralNet::CrossEntropyCost<float>, NeuralNet::ReLUActivation<float>>;
using NeuralNet2 = NeuralNet::Network<float, NeuralNet::CrossEntropyCost<float>, NeuralNet::SigmoidActivation<float>>;
using NeuralNet3 = NeuralNet::Network<float, NeuralNet::CrossEntropyCost<float>, NeuralNet::TanhActivation<float>>;
using NeuralNet4 = NeuralNet::Network<float, NeuralNet::QuadraticCost<float, NeuralNet::ReLUActivation<float>>, NeuralNet::ReLUActivation<float>>;
using NeuralNet5 = NeuralNet::Network<float, NeuralNet::QuadraticCost<float, NeuralNet::ReLUActivation<float>>, NeuralNet::ReLUActivation<float>>;
using NeuralNet6 = NeuralNet::Network<float, NeuralNet::QuadraticCost<float, NeuralNet::TanhActivation<float>>, NeuralNet::TanhActivation<float>>;

using NetData = std::vector<NeuralNet1::TrainingData>;
using NetDataMem = std::vector<std::pair<std::vector<float>, std::vector<float>>>;
NetData td;
NetData testData;
NetDataMem training_data_mem;
NetDataMem test_data_mem;

namespace NeuralTest
{
    std::filesystem::path data_path;
    void LoadData()
    {
        const path t10K_images = data_path / path("data") / path("t10k-images.idx3-ubyte");
        const path t10K_labels = data_path / path("data") / path("t10k-labels.idx1-ubyte");
        const path train_images = data_path / path("data") / path("train-images.idx3-ubyte");
        const path train_labels = data_path / path("data") / path("train-labels.idx1-ubyte");
        try {
            // Load training data
            mnist_loader<float> loader(train_images, train_labels, training_data_mem, td);
            // Load test data
            mnist_loader<float> loader2(t10K_images, t10K_labels, test_data_mem, testData);
        }
        catch (const char* Error) {
            std::cout << "Error: " << Error << "\n"
                << train_images << "\n"
                << train_labels << "\n"
                << t10K_images << "\n"
                << t10K_labels << "\n";
        }
    }

    bool IsSame(const NeuralNet1& net, int test_case)
    {
        std::vector<float> res(10);
        mdspan<float, std::dextents<size_t, 2>> nres(res.data(), (size_t)res.size(), 1);
        net.feedforward(testData[test_case].first, nres);
        auto x = net.result(nres);
        auto y = net.result(testData[test_case].second);
        return x == y;
    }

    void LoadFromFile(NeuralNet1& net, std::filesystem::path fileName)
    {
        std::fstream f(fileName, std::ios::binary | std::ios::in);
        if (!f.is_open()) {
            std::cout << "failed to open " << fileName << "\n";
            return;
        }
        else {
            f >> net;
        }
    }

    TEST(ReLUActivationCrossEntropyCost, TestFeedForward) {
        NeuralNet1 net;
        LoadFromFile(net, data_path / path("data") / path("net-save.txt"));

        int test_case = 0;
        EXPECT_TRUE(IsSame(net, 0));
    }

    TEST(ReLUActivationCrossEntropyCost, TestTraining) {
        NeuralNet1 net({ 784, 100, 10 });
        float Lmbda = 0.1f;
        float eta = 0.03f;
        net.SGD(td.begin(), td.begin() + 100, 1, 100, eta, Lmbda,
            [](const NeuralNet1& network, int Epoch, float& currenctEta) {
                // eta can be manipulated in the feed back function
            });

        auto accuracy = net.accuracy(testData.begin(), testData.begin() + 100);
        //std::cout << accuracy << "\n";
        // anything > 0 is ok
        EXPECT_TRUE(accuracy > 1);
    }

    TEST(SigmoidActivationCrossEntropyCost, TestTraining) {
        NeuralNet2 net({ 784, 100, 10 });
        float Lmbda = 0.09f;
        float eta = 0.04f;
        net.SGD(td.begin(), td.begin() + 100, 1, 100, eta, Lmbda,
            [](const NeuralNet2& network, int Epoch, float& currenctEta) {
                // eta can be manipulated in the feed back function
            });

        auto accuracy = net.accuracy(testData.begin(), testData.begin() + 100);
        //std::cout << accuracy << "\n";
        // Anything > 0
        EXPECT_TRUE(accuracy > 1);
    }

    TEST(TanhActivationCrossEntropyCost, TestTraining) {
        NeuralNet3 net({ 784, 100, 10 });
        float Lmbda = 0.09f;
        float eta = 0.04f;
        net.SGD(td.begin(), td.begin() + 100, 1, 100, eta, Lmbda,
            [](const NeuralNet3& network, int Epoch, float& currenctEta) {
                // eta can be manipulated in the feed back function
            });

        auto accuracy = net.accuracy(testData.begin(), testData.begin() + 100);
        //std::cout << accuracy << "\n";
        // Anything > 0
        EXPECT_TRUE(accuracy > 1);
    }

    TEST(ReLUActivationQuadraticCost, TestTraining) {
        NeuralNet4 net({ 784, 100, 10 });
        float Lmbda = 0.1f;
        float eta = 0.03f;
        net.SGD(td.begin(), td.begin() + 100, 1, 100, eta, Lmbda,
            [](const NeuralNet4& network, int Epoch, float& currenctEta) {
                // eta can be manipulated in the feed back function
            });

        auto accuracy = net.accuracy(testData.begin(), testData.begin() + 100);
        //std::cout << accuracy << "\n";
        // anything > 0 is ok
        EXPECT_TRUE(accuracy > 1);
    }

    TEST(SigmoidActivationQuadraticCost, TestTraining) {
        NeuralNet5 net({ 784, 100, 10 });
        float Lmbda = 0.09f;
        float eta = 0.04f;
        net.SGD(td.begin(), td.begin() + 100, 1, 100, eta, Lmbda,
            [](const NeuralNet5& network, int Epoch, float& currenctEta) {
                // eta can be manipulated in the feed back function
            });

        auto accuracy = net.accuracy(testData.begin(), testData.begin() + 100);
        //std::cout << accuracy << "\n";
        // Anything > 0
        EXPECT_TRUE(accuracy > 1);
    }

    TEST(TanhActivationQuadraticCost, TestTraining) {
        NeuralNet6 net({ 784, 100, 10 });
        float Lmbda = 0.09f;
        float eta = 0.04f;
        net.SGD(td.begin(), td.begin() + 100, 1, 100, eta, Lmbda,
            [](const NeuralNet6& network, int Epoch, float& currenctEta) {
                // eta can be manipulated in the feed back function
            });

        auto accuracy = net.accuracy(testData.begin(), testData.begin() + 100);
        //std::cout << accuracy << "\n";
        // Anything > 0
        EXPECT_TRUE(accuracy > 1);
    }

    TEST(NeuralNet, TestAssignements) {
        NeuralNet1 net;
        LoadFromFile(net, data_path / path("data") / path("net-save.txt"));
        // test assigments

        EXPECT_TRUE(IsSame(net, 0));
        NeuralNet1 net2(std::move(net));
        EXPECT_TRUE(IsSame(net2, 1));

        NeuralNet1 net3{};
        net3 = std::move(net2);
        EXPECT_TRUE(IsSame(net3, 2));

        NeuralNet1 net4(net3);
        net3 = net4;
        EXPECT_TRUE(IsSame(net4, 3));
        EXPECT_TRUE(IsSame(net3, 4));
    }

}

using namespace NeuralTest;

GTEST_API_ int main(int argc, char** argv) {
    std::cout << "Running main() from " << __FILE__ << "\n";
    NeuralTest::data_path = path(argv[0]).parent_path().parent_path();
    std::cout << "Data directory: " << NeuralTest::data_path << "\n";
    LoadData();
    std::cout << "Loaded training data " << argv[0] << "\n";
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
