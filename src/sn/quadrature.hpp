#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

dealii::Quadrature<2> gauss_chebyshev(int order);

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate);

dealii::Quadrature<2> impose_polar_symmetry(
    const dealii::Quadrature<2> &quadrature);

template <int dim>
dealii::Quadrature<dim> reorder(const dealii::Quadrature<dim> &quadrature);

template <int dim>
struct CompareQuadraturePoints {
  CompareQuadraturePoints(const std::vector<dealii::Point<dim>> &points)
      : points(points) {};
  bool operator() (const int a, const int b) const {
    const double pol_a = points[a](0);
    const double pol_b = points[b](0);
    Assert(pol_a != 0.5 && pol_b != 0.5, dealii::ExcInvalidState());
    if (dim == 2) {
      const double azi_a = points[a](1);
      const double azi_b = points[b](1);
      Assert(azi_a != 0.5 && azi_b != 0.5, dealii::ExcInvalidState());
      if (azi_a != azi_b) {
        return azi_a < azi_b;  // sort by azimuthal
      } else {
        if (azi_a < 0.5)
          return pol_a < pol_b;  // sort equal azimuthal < pi by polar
        else
          return pol_a > pol_b;  // sort equal azimuthal > pi by reverse polar
      }
    } else if (dim == 1) {
      if (pol_a > 0.5 && pol_b > 0.5) 
        return pol_a < pol_b;  // sort by polar
      else if (pol_a < 0.5 && pol_a < 0.5)
        return pol_a > pol_b;  // sort by reverse polar
      else
        return pol_a < pol_b;  // sort by octant
    } else {
      throw dealii::ExcNotImplemented();
    }
  };
  const std::vector<dealii::Point<dim>> points;
};

#endif  // AETHER_SN_QUADRATURE_H_