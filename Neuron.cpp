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

} // namespace Sagacity