#define BOOST_TEST_MODULE NetworkTests
#define BOOST_TEST_MAIN

#include "../Network.hpp"
#include <boost/test/unit_test.hpp>
#include <fstream>

using namespace Sagacity;

BOOST_AUTO_TEST_CASE(orNetwork) {

  Network network(2, {{1, ActivationFunction::HardLim}});

  Network::TrainingData trainingData = {
      {{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {1}}};

  network.train(trainingData, 100);

  for (const std::pair<Vector, Vector> &trainingSet : trainingData)
    BOOST_CHECK_EQUAL(network(trainingSet.first), trainingSet.second);
}

BOOST_AUTO_TEST_CASE(andNetwork) {

  Network network(2, {{1, ActivationFunction::HardLim}});

  Network::TrainingData trainingData = {
      {{0, 0}, {0}}, {{0, 1}, {0}}, {{1, 0}, {0}}, {{1, 1}, {1}}};

  network.train(trainingData, 100);

  for (const std::pair<Vector, Vector> &trainingSet : trainingData)
    BOOST_CHECK_EQUAL(network(trainingSet.first), trainingSet.second);
}

BOOST_AUTO_TEST_CASE(exOrNetwork) {

  Network network(
      2, {{2, ActivationFunction::TanSig}, {1, ActivationFunction::HardLim}});

  Network::TrainingData trainingData = {
      {{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};

  network.train(trainingData, 500, 1);

  for (const std::pair<Vector, Vector> &trainingSet : trainingData)
    BOOST_CHECK_EQUAL(network(trainingSet.first), trainingSet.second);
}

BOOST_AUTO_TEST_CASE(saveAndLoad) {

  Network::TrainingData trainingData = {
      {{1, 2, 3}, {3, 5}}, {{2, 3, 4}, {5, 7}}, {{3, 4, 5}, {7, 9}}};

  std::vector<Vector> oldOutputs(3), newOutputs(3);

  {
    Network network(
        3, {{5, ActivationFunction::TanSig}, {2, ActivationFunction::PureLin}});
    network.train(trainingData, 100);
    std::transform(trainingData.cend(), trainingData.cend(), oldOutputs.begin(),
                   [&network](const std::pair<Vector, Vector> &data) {
                     return network(data.first);
                   });
    std::ofstream fileOut("test.json");
    fileOut << network.toJSON();
    fileOut.close();
  }

  {
    std::ifstream fileIn("test.json");
    nlohmann::json json;
    fileIn >> json;
    fileIn.close();
    Network network(json);
    std::transform(trainingData.cend(), trainingData.cend(), newOutputs.begin(),
                   [&network](const std::pair<Vector, Vector> &data) {
                     return network(data.first);
                   });
  }

  for (size_t i = 0; i < 3; i++)
    BOOST_CHECK_EQUAL(oldOutputs[i], newOutputs[i]);
}