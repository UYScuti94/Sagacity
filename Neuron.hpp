#pragma once

#include "ActivationFunction.hpp"
#include "Vector.hpp"
#include <list>
#include <random>

namespace Sagacity {

class Neuron {

private:
  friend class Layer;

  Vector m_weights, m_deltaWeights;
  mutable Vector m_lastInputs;
  mutable double m_lastSum;
  double m_bias, m_deltaBias = 0;
  const ActivationFunction &m_activationFunction;
  double m_error = 0;
  bool m_training = false;

  static class RandomNumberGenerator {
  private:
    std::random_device rd;
    std::mt19937 twister;
    std::uniform_real_distribution<> generator;

  public:
    RandomNumberGenerator() : twister(rd()), generator(-1, 1) {}
    inline double next() { return generator(twister); }
  } m_randomNumberGenerator;

  double operator()(const Vector &inputs) const;
  Vector backPropError() const;
  void correctWeights(double alpha, double gamma);

public:
  Neuron(size_t numberOfInputs, const ActivationFunction &activationFunction);
};

} // namespace Sagacity