#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

dealii::Quadrature<2> gauss_chebyshev(int order);

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate) {
  dealii::Point<dim> ordinate;
  if (qdim == 1) {
    Assert(dim == 1, dealii::ExcInvalidState());
    ordinate(0) = 2 * coordinate(0) - 1;
  }
  else {
    Assert(qdim == 2, dealii::ExcInvalidState());
    Assert(2 <= dim && dim <= 3, dealii::ExcInvalidState());
    ordinate(0) = coordinate(0);
    double theta = std::acos(coordinate(0));
    double phi = std::acos(coordinate(1));
    ordinate(1) = std::sin(theta) * std::cos(phi);
    if (dim == 3)
      ordinate(2) = std::sin(theta) * std::sin(phi);
  }
  return ordinate;
}

dealii::Quadrature<2> impose_azimuthal_symmetry(
    const dealii::Quadrature<2> &quadrature);

#endif  // AETHER_SN_QUADRATURE_H_