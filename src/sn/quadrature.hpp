#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

namespace aether::sn {

dealii::Quadrature<2> gauss_chebyshev(int order);

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate);

dealii::Quadrature<1> impose_polar_symmetry(
    const dealii::Quadrature<1> &quadrature);

template <int dim>
dealii::Quadrature<dim> reorder(const dealii::Quadrature<dim> &quadrature);

template <int dim>
struct CompareQuadraturePoints {
  CompareQuadraturePoints(const std::vector<dealii::Point<dim>> &points);
  bool operator() (const int a, const int b) const;
  const std::vector<dealii::Point<dim>> points;
};

}  // namespace aether::sn

#endif  // AETHER_SN_QUADRATURE_H_