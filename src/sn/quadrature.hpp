#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>

dealii::Quadrature<2> gauss_chebyshev(int order) {
  int num_points = 2 * order - 1;
  dealii::QGauss<1> polar(num_points);
  dealii::QGaussChebyshev<1> azimuthal(num_points);
  return dealii::Quadrature<2>(polar, azimuthal);
};

template <int dim>
dealii::Tensor<1, dim> ordinate(Point<2> coordinate) {
  Assert(2 <= dim <= 3, dealii::ExcMessage("2D or 3D only"))
  dealii::Tensor<1, dim> ordinate;
  double theta = std::acos(coordinate(0));
  double phi = std::acos(coordinate(1));
  ordinate(0) = std::cos(theta);
  ordinate(1) = std::sin(theta) * std::cos(phi);
  if (dim == 3)
    ordinate(2) = std::sin(theta) * std::sin(phi);
  return ordinate;
};

dealii::Tensor<1, 1> ordinate(dealii::Point<1> coordinate) {
  return coordinate;
};

#endif  // AETHER_SN_QUADRATURE_H_