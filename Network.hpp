#pragma once

#include "Layer.hpp"

namespace Sagacity {

class Network {

private:
  std::list<Layer> m_layers;

  void setErrors(Vector errors);
  void setTraining(bool training);
  void correctWeights(double alpha, double gamma);
  void trainingCycle(const Vector &inputs, const Vector &targets, double alpha,
                     double gamma);

public:
  typedef std::pair<size_t, std::reference_wrapper<const ActivationFunction>>
      LayerConfig;

  typedef std::initializer_list<LayerConfig> NetworkConfig;

  Network(size_t numberOfInputs, const NetworkConfig &networkConfig);

  Vector operator()(Vector inputs) const;

  typedef std::vector<std::pair<Vector, Vector>> TrainingData;

  void train(const TrainingData &trainingData, size_t numberOfCycles,
             double alpha = .1, double gamma = .1);
};

} // namespace Sagacity