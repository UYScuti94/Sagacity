#pragma once

#include <ostream>
#include <vector>

namespace Sagacity {

class Vector : public std::vector<double> {

public:
  Vector() = default;
  Vector(size_t size);
  Vector(const std::initializer_list<double> &data);

  Vector operator*(double factor) const;

  Vector operator+(const Vector &other) const;
  Vector operator-(const Vector &other) const;
  Vector operator*(const Vector &other) const;

  Vector &operator+=(const Vector &other);
  Vector &operator*=(const Vector &other);

  double dotProduct(const Vector &other, double init = 0) const;
};

inline Vector operator*(double factor, const Vector &vector) {
  return vector * factor;
}

std::ostream &operator<<(std::ostream &stream, const Vector &vector);

} // namespace Sagacity