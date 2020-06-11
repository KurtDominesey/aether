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

template <int dim, int qdim>
int DiscreteToMoment<dim, qdim>::n_block_rows(int order) const {
  switch (dim) {
    case 1: return order + 1;
    case 2: return (order + 1) * (order + 2) / 2;
    case 3: return std::pow(order+1, 2);
  }
  AssertThrow(false, dealii::ExcImpossibleInDim(dim));
}

template <int dim, int qdim>
int DiscreteToMoment<dim, qdim>::n_block_cols() const {
  return quadrature.size();
}

template <int dim, int qdim>
void DiscreteToMoment<dim, qdim>::discrete_to_legendre(
    dealii::Vector<double> &dst, const dealii::Vector<double> &src) const {
  const int num_dofs = src.size() / quadrature.size();
  const int num_ell = dst.size() / num_dofs;
  AssertThrow(src.size() % quadrature.size() == 0, 
              dealii::ExcNotMultiple(src.size(), quadrature.size()));
  AssertThrow(dst.size() % num_dofs == 0,
              dealii::ExcNotMultiple(dst.size(), num_dofs));
  dealii::BlockVector<double> dst_b(num_ell, num_dofs);
  dealii::BlockVector<double> src_b(quadrature.size(), num_dofs);
  src_b = src;
  discrete_to_legendre(dst_b, src_b);
  dst = dst_b;
}

template <int dim, int qdim>
void DiscreteToMoment<dim, qdim>::discrete_to_legendre(
    dealii::BlockVector<double> &dst, const dealii::BlockVector<double> &src)
    const {
  dst = 0;
  const int order = dst.n_blocks() - 1;
  const int num_moments = n_block_rows(order);
  const int num_dofs = dst.block(0).size();
  dealii::BlockVector<double> moments(num_moments, num_dofs);
  vmult(moments, src);
  for (int ell = 0, lm = 0; ell <= order; ++ell) {
    switch (dim) {
      case 1:
        dst.block(ell) += moments.block(ell);
        break;
      case 2:
        for (int m = -ell; m <= -ell; m += 2, ++lm)
          dst.block(ell) += moments.block(lm);
        break;
      case 3:
        for (int m = -ell; m <= -ell; ++m, ++lm)
          dst.block(ell) += moments.block(lm);
        break;
    }
  }
}

template class DiscreteToMoment<1>;
template class DiscreteToMoment<2>;
template class DiscreteToMoment<3>;

}  // namespace aether::sn