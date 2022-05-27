#pragma once

#include "Neuron.hpp"
#include <list>

namespace Sagacity {

class Layer {

  friend class Network;

private:
  std::list<Neuron> m_neurons;

  Vector operator()(const Vector &inputs) const;
  void setErrors(const Vector &errors);
  Vector backPropErrors() const;
  void correctWeights(double alpha, double gamma);
  void setTraining(bool training);

  nlohmann::json toJSON() const;

public:
  Layer(size_t numberOfInputs, size_t numberOfNeurons,
        const ActivationFunction &activationFunction);
  Layer(const nlohmann::json &json);
};

} // namespace Sagacity