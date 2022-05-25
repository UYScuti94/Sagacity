#define BOOST_TEST_MODULE NetworkTests
#define BOOST_TEST_MAIN

#include "../Network.hpp"
#include <boost/test/unit_test.hpp>

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