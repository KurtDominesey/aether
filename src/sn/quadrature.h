#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

namespace aether::sn {

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate);

dealii::Quadrature<1> impose_polar_symmetry(
    const dealii::Quadrature<1> &quadrature);

}  // namespace aether::sn

#endif  // AETHER_SN_QUADRATURE_H_