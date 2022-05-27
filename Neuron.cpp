#include "Neuron.hpp"
#include <algorithm>
#include <cassert>

namespace Sagacity {

Neuron::RandomNumberGenerator Neuron::m_randomNumberGenerator;

Neuron::Neuron(size_t numberOfInputs,
               const ActivationFunction &activationFunction)
    : m_weights(numberOfInputs), m_deltaWeights(numberOfInputs),
      m_activationFunction(activationFunction) {
  std::transform(m_weights.cbegin(), m_weights.cend(), m_weights.begin(),
                 [](double) { return m_randomNumberGenerator.next(); });
  m_bias = m_randomNumberGenerator.next();
}

Neuron::Neuron(const nlohmann::json &json)
    : m_weights(json["weights"].size()), m_deltaWeights(json["weights"].size()),
      m_activationFunction(
          ActivationFunction::called(json["activationFunction"])),
      m_bias(json["bias"]) {
  size_t i = 0;
  std::transform(m_weights.cbegin(), m_weights.cend(), m_weights.begin(),
                 [&i, &json](double) { return json["weights"][i++]; });
}

double Neuron::operator()(const Vector &inputs) const {
  if (m_training)
    m_lastInputs = inputs;
  m_lastSum = m_weights.dotProduct(inputs, m_bias);
  return m_activationFunction(m_lastSum);
}

Vector Neuron::backPropError() const { return m_weights * m_error; }

void Neuron::correctWeights(double alpha, double gamma) {
  m_deltaWeights = -alpha * m_error *
                       m_activationFunction.derivative(m_lastSum) *
                       m_lastInputs +
                   m_deltaWeights * gamma;
  m_weights += m_deltaWeights;
  m_deltaBias = -alpha * m_error + m_deltaBias * gamma;
  m_bias += m_deltaBias;
}

nlohmann::json Neuron::toJSON() const {
  return {
      {"weights", m_weights},
      {"bias", m_bias},
      {"activationFunction", ActivationFunction::name(m_activationFunction)}};
}

} // namespace Sagacity