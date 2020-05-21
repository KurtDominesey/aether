#include "discrete_to_moment.h"

namespace aether::sn {

template <int dim, int qdim>
DiscreteToMoment<dim, qdim>::DiscreteToMoment(
    const QAngle<dim, qdim> &quadrature)
    : quadrature(quadrature) {}

template <int dim, int qdim>
void DiscreteToMoment<dim, qdim>::vmult(dealii::Vector<double> &dst,
                                        const dealii::Vector<double> &src) 
                                        const {
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int qdim>
void DiscreteToMoment<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                        const dealii::BlockVector<double> &src) 
                                        const {
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int qdim>
void DiscreteToMoment<dim, qdim>::vmult_add(dealii::Vector<double> &dst,
                                            const dealii::Vector<double> &src) 
                                            const {
  const int num_ords = quadrature.size();
  const int num_dofs = src.size() / num_ords;
  const int num_moments = dst.size() / num_dofs;
  dealii::BlockVector<double> dst_b(num_moments, num_dofs);
  dealii::BlockVector<double> src_b(num_ords, num_dofs);
  dst_b = dst;
  src_b = src;
  vmult_add(dst_b, src_b);
  dst = dst_b;
}

template <int dim, int qdim>
void DiscreteToMoment<dim, qdim>::vmult_add(
    dealii::BlockVector<double> &dst, const dealii::BlockVector<double> &src) 
    const {
  int num_moments = dst.n_blocks();
  int num_ordinates = src.n_blocks();
  AssertDimension(num_ordinates, quadrature.size());
  for (int n = 0; n < quadrature.size(); ++n) {
    for (int ell = 0, lm = 0; lm < num_moments; ++ell) {
      switch (dim) {
        case 1:
          dst.block(ell).add(quadrature.weight(n) *
                             spherical_harmonic(ell, 0, quadrature.angle(n)),
                             src.block(n));
          ++lm;  // lm == ell
          break;
        case 2:
          for (int m = -ell; m <= ell; m += 2, ++lm)
            dst.block(lm).add(quadrature.weight(n) *
                              spherical_harmonic(ell, m, quadrature.angle(n)),
                              src.block(n));
          break;
        case 3:
          for (int m = -ell; m <= ell; ++m, ++lm)
            dst.block(lm).add(quadrature.weight(n) *
                              spherical_harmonic(ell, m, quadrature.angle(n)),
                              src.block(n));
          break;
      }
    }
  }
}

template class DiscreteToMoment<1>;
template class DiscreteToMoment<2>;
template class DiscreteToMoment<3>;

}  // namespace aether::sn