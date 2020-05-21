#ifndef AETHER_SN_SPHERICAL_HARMONICS_H_
#define AETHER_SN_SPHERICAL_HARMONICS_H_

#include <deal.II/base/point.h>
#include <deal.II/lac/block_vector.h>

#include "sn/discrete_to_moment.h"

namespace aether::sn {

template <int qdim>
double spherical_harmonic(const int ell, const int m, 
                          const dealii::Point<qdim>& angle) {
  AssertThrow(std::abs(m) <= ell, dealii::ExcIndexRange(m, -ell, ell+1));
  double polar = angle[0];
  double azim = std::nan("a");
  switch (qdim) {
    case 1: AssertThrow(m == 0, dealii::ExcImpossibleInDim(qdim)); break;
    case 2: azim = angle[1]; break;
    default: throw dealii::ExcNotImplemented();
  }
  // hardcoded functions (TODO: implement higher orders)
  switch (ell) {
    case 0:
      return 1;
    case 1:
      switch (m) {
        case -1:
          return -std::sqrt(1-std::pow(polar, 2)) * std::sin(azim);
        case 0:
          return polar;
        case +1:
          return -std::sqrt(1-std::pow(polar, 2)) * std::cos(azim);
      }
  }
  throw dealii::ExcInvalidState();  // should not reach here
}

}  // namespace aether::sn

#endif  // AETHER_SN_SPHERICAL_HARMONICS_H_