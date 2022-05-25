#include "Vector.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>

namespace Sagacity {

Vector::Vector(size_t size) : std::vector<double>(size) {}

Vector::Vector(const std::initializer_list<double> &data)
    : std::vector<double>(data) {}

Vector Vector::operator*(double factor) const {
  Vector product(size());
  std::transform(cbegin(), cend(), product.begin(),
                 [factor](double element) { return element * factor; });
  return product;
}

Vector Vector::operator+(const Vector &other) const {
  assert(size() == other.size());
  Vector sum(size());
  std::transform(cbegin(), cend(), other.cbegin(), sum.begin(),
                 std::plus<double>());
  return sum;
}

Vector Vector::operator-(const Vector &other) const {
  assert(size() == other.size());
  Vector diff(size());
  std::transform(cbegin(), cend(), other.cbegin(), diff.begin(),
                 std::minus<double>());
  return diff;
}

Vector Vector::operator*(const Vector &other) const {
  assert(size() == other.size());
  Vector product(size());
  std::transform(cbegin(), cend(), other.cbegin(), product.begin(),
                 std::multiplies<double>());
  return product;
}

Vector &Vector::operator+=(const Vector &other) {
  std::transform(cbegin(), cend(), other.cbegin(), begin(),
                 std::plus<double>());
  return *this;
}

Vector &Vector::operator*=(const Vector &other) {
  std::transform(cbegin(), cend(), other.cbegin(), begin(),
                 std::multiplies<double>());
  return *this;
}

double Vector::dotProduct(const Vector &other, double init) const {
  assert(size() == other.size());
  return std::inner_product(cbegin(), cend(), other.cbegin(), init);
}

std::ostream &operator<<(std::ostream &stream, const Vector &vector) {
  Vector::const_iterator it = vector.cbegin();
  stream << "[ " << *it;
  for_each(++it, vector.cend(),
           [&stream](double value) { stream << ", " << value; });
  stream << " ]" << std::endl;
  return stream;
}

} // namespace Sagacity