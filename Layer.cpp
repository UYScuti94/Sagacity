#include "Layer.hpp"
#include <algorithm>
#include <assert.h>

namespace Sagacity {

Layer::Layer(size_t numberOfInputs, size_t numberOfNeurons,
             const ActivationFunction &activationFunction) {
  for (size_t i = 0; i < numberOfNeurons; i++)
    m_neurons.emplace_back(numberOfInputs, activationFunction);
}

Vector Layer::operator()(const Vector &inputs) const {
  Vector outputs(m_neurons.size());
  std::transform(m_neurons.cbegin(), m_neurons.cend(), outputs.begin(),
                 [&inputs](const Neuron &neuron) { return neuron(inputs); });
  return outputs;
}

void Layer::setErrors(const Vector &errors) {
  assert(m_neurons.size() == errors.size());
  size_t i = 0;
  for (Neuron &neuron : m_neurons)
    neuron.m_error = errors[i++];
}

Vector Layer::backPropErrors() const {
  std::list<Neuron>::const_iterator it = m_neurons.cbegin();
  Vector errors = it->backPropError();
  std::for_each(++it, m_neurons.cend(), [&errors](const Neuron &neuron) {
    errors += neuron.backPropError();
  });
  return errors;
}

void Layer::correctWeights(double alpha, double gamma) {
  for (Neuron &neuron : m_neurons)
    neuron.correctWeights(alpha, gamma);
}

void Layer::setTraining(bool training) {
  for (Neuron &neuron : m_neurons)
    neuron.m_training = training;
}

} // namespace Sagacity