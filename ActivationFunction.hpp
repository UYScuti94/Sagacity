#pragma once

#include <functional>

namespace Sagacity {

class ActivationFunction {

private:
  friend class Neuron;

  std::function<double(double)> m_activationFunction;
  std::function<double(double)> m_derivative;

  ActivationFunction(const std::function<double(double)> &activationFunction,
                     const std::function<double(double)> &derivative);

  inline double operator()(double x) const { return m_activationFunction(x); }
  inline double derivative(double x) const { return m_derivative(x); }

public:
  ActivationFunction(const ActivationFunction &) = delete;
  ActivationFunction(ActivationFunction &&) = delete;

  ActivationFunction &operator=(const ActivationFunction &) = delete;
  ActivationFunction &operator=(ActivationFunction &&) = delete;

  static const ActivationFunction HardLim;
  static const ActivationFunction HardLims;
  static const ActivationFunction HardLimss;
  static const ActivationFunction PureLin;
  static const ActivationFunction ReLU;
  static const ActivationFunction LogSig;
  static const ActivationFunction TanSig;
};

} // namespace Sagacity