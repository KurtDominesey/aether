#include "discrete_to_moment.hpp"

template <int qdim>
DiscreteToMoment<qdim>::DiscreteToMoment(
    const dealii::Quadrature<qdim> &quadrature)
    : quadrature(quadrature) {}

template <int qdim>
void DiscreteToMoment<qdim>::vmult(dealii::BlockVector<double> &dst,
                                   const dealii::BlockVector<double> &src) 
                                   const {
  dst = 0;
  vmult_add(dst, src);
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

template <int qdim>
void DiscreteToMoment<qdim>::Tvmult(dealii::BlockVector<double> &dst,
                                   const dealii::BlockVector<double> &src) 
                                   const {} // not implemented

template <int qdim>
void DiscreteToMoment<qdim>::Tvmult_add(dealii::BlockVector<double> &dst,
                                        const dealii::BlockVector<double> &src) 
                                        const {} // not implemented

template class DiscreteToMoment<1>;
template class DiscreteToMoment<2>;