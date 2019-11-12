#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

dealii::Quadrature<2> gauss_chebyshev(int order);

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate);

dealii::Quadrature<2> impose_azimuthal_symmetry(
    const dealii::Quadrature<2> &quadrature);

#endif  // AETHER_SN_QUADRATURE_H_