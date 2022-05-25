#include "ActivationFunction.hpp"
#include <cmath>

namespace Sagacity {

ActivationFunction::ActivationFunction(
    const std::function<double(double)> &activationFunction,
    const std::function<double(double)> &derivative)
    : m_activationFunction(activationFunction), m_derivative(derivative) {}

static double return1(double) { return 1; }

const ActivationFunction
    ActivationFunction::HardLim([](double x) { return x >= 0; }, return1);

const ActivationFunction ActivationFunction::HardLims(
    [](double x) { return x > 0   ? 1
                          : x < 0 ? -1
                                  : 0; }, return1);

const ActivationFunction ActivationFunction::HardLimss(
    [](double x) { return x >= 1    ? 1
                          : x <= -1 ? -1
                                    : 0; }, return1);

const ActivationFunction ActivationFunction::PureLin([](double x) { return x; },
                                                     return1);

const ActivationFunction
    ActivationFunction::ReLU([](double x) { return x > 0 ? x : 0; }, return1);

static double logsig(double x) { return 1.0 / (1 + exp(-x)); }

const ActivationFunction ActivationFunction::LogSig(logsig, [](double x) {
  double y = logsig(x);
  return y * (1.0 - y);
});

static double tansig(double x) { return 2.0 / (1 + exp(-2 * x)) - 1; }

const ActivationFunction ActivationFunction::TanSig(tansig, [](double x) {
  return 1.0 - pow(tansig(x), 2);
});

} // namespace Sagacity