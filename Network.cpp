#include "Network.hpp"
#include <algorithm>
#include <cassert>

namespace Sagacity {

Network::Network(size_t numberOfInputs, const NetworkConfig &networkConfig) {
  for (const LayerConfig &layerConfig : networkConfig) {
    m_layers.emplace_back(numberOfInputs, layerConfig.first,
                          layerConfig.second);
    numberOfInputs = layerConfig.first;
  }
}

Network::Network(const nlohmann::json &json) {
  for (size_t i = 0; i < json["layers"].size(); i++)
    m_layers.emplace_back(json["layers"][i]);
}

Vector Network::operator()(Vector inputs) const {
  Vector &outputs = inputs;
  for (const Layer &layer : m_layers)
    outputs = layer(inputs);
  return outputs;
}

void Network::setErrors(Vector errors) {
  std::for_each(m_layers.rbegin(), m_layers.rend(), [&errors](Layer &layer) {
    layer.setErrors(errors);
    errors = layer.backPropErrors();
  });
}

void Network::setTraining(bool training) {
  for (Layer &layer : m_layers)
    layer.setTraining(training);
}

void Network::correctWeights(double alpha, double gamma) {
  for (Layer &layer : m_layers)
    layer.correctWeights(alpha, gamma);
}

void Network::trainingCycle(const Vector &inputs, const Vector &targets,
                            double alpha, double gamma) {
  const Vector &outputs = operator()(inputs);
  const Vector &errors = outputs - targets;
  setErrors(errors);
  correctWeights(alpha, gamma);
}

void Network::train(const TrainingData &trainingData, size_t numberOfCycles,
                    double alpha, double gamma) {
  setTraining(true);
  for (size_t i = 0; i < numberOfCycles; i++)
    for (const auto &trainingPair : trainingData)
      trainingCycle(trainingPair.first, trainingPair.second, alpha, gamma);
  setTraining(false);
}

nlohmann::json Network::toJSON() const {
  nlohmann::json network;
  size_t i = 0;
  for (const Layer &layer : m_layers)
    network["layers"][i++] = layer.toJSON();
  return network;
}

} // namespace Sagacity