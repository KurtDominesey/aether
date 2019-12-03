#include "discrete_to_moment.hpp"

template <int qdim>
DiscreteToMoment<qdim>::DiscreteToMoment(
    const dealii::Quadrature<qdim> &quadrature)
    : quadrature(quadrature) {}

template <int qdim>
void DiscreteToMoment<qdim>::vmult(dealii::Vector<double> &dst,
                                   const dealii::Vector<double> &src) const {
  const int num_ords = quadrature.size();
  const int num_dofs = src.size() / num_ords;
  dealii::BlockVector<double> dst_b(1, num_dofs);
  dealii::BlockVector<double> src_b(num_ords, num_dofs);
  src_b = src;
  vmult(dst_b, src_b);
  dst = dst_b;
}

template <int qdim>
void DiscreteToMoment<qdim>::vmult(dealii::BlockVector<double> &dst,
                                   const dealii::BlockVector<double> &src) 
                                   const {
  dst = 0;
  vmult_add(dst, src);
}

template <int qdim>
void DiscreteToMoment<qdim>::vmult_add(dealii::Vector<double> &dst,
                                       const dealii::Vector<double> &src) 
                                       const {
  const int num_ords = quadrature.size();
  const int num_dofs = src.size() / num_ords;
  dealii::BlockVector<double> dst_b(1, num_dofs);
  dealii::BlockVector<double> src_b(num_ords, num_dofs);
  dst_b = dst;
  src_b = src;
  vmult_add(dst_b, src_b);
  dst = dst_b;
}

template <int qdim>
void DiscreteToMoment<qdim>::vmult_add(dealii::BlockVector<double> &dst,
                                       const dealii::BlockVector<double> &src) 
                                       const {
  int num_moments = dst.n_blocks();
  int num_ordinates = src.n_blocks();
  Assert(num_moments == 1, dealii::ExcNotImplemented());
  Assert(num_ordinates == quadrature.get_weights().size(),
         dealii::ExcDimensionMismatch(num_ordinates,
                                      quadrature.get_weights().size()));
  for (int ell = 0, lm = 0; lm < num_moments; ++ell) {
    for (int m = -ell; m <= ell; ++m, ++lm) {
      // Y_lm = std::bind(spherical_harmonics(ell, m, _1, _2))
      for (int n = 0; n < num_ordinates; ++n) {
        dst.block(lm).add(quadrature.weight(n) * 1.0, src.block(n));
      }
    }
  }
}

template class DiscreteToMoment<1>;
template class DiscreteToMoment<2>;