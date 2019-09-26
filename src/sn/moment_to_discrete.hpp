#ifndef AETHER_SN_MOMENT_TO_DISCRETE_H_
#define AETHER_SN_MOMENT_TO_DISCRETE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/block_vector.h>

template <int qdim>
class MomentToDiscrete {
 public:
  MomentToDiscrete(const dealii::Quadrature<qdim> &quadrature);
  void vmult(dealii::BlockVector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;

 protected:
  const dealii::Quadrature<qdim> &quadrature;
};

template <int qdim>
MomentToDiscrete<qdim>::MomentToDiscrete(
    const dealii::Quadrature<qdim> &quadrature)
    : quadrature(quadrature) {}

template <int qdim>
void MomentToDiscrete<qdim>::vmult(dealii::BlockVector<double> &dst,
                                   const dealii::BlockVector<double> &src) 
                                   const {
  dst = 0;
  int num_ordinates = dst.n_blocks();
  int num_moments = src.n_blocks();
  Assert(num_moments == 1, dealii::ExcNotImplemented());
  Assert(num_ordinates == quadrature.get_points().size(),
         dealii::ExcDimensionMismatch(num_ordinates,
                                      quadrature.get_points().size()));
  for (int ell = 0, lm = 0; lm < num_moments; ++ell) {
    for (int m = -ell; m <= ell; ++m, ++lm) {
      // Y_lm = std::bind(spherical_harmonics(ell, m, _1, _2));
      for (int n = 0; n < num_ordinates; ++n) {
        dst.block(n).add(1.0, src.block(lm));
      }
    }
  }
}

template <int qdim>
void MomentToDiscrete<qdim>::Tvmult(dealii::BlockVector<double> &dst,
                                   const dealii::BlockVector<double> &src) 
                                   const {} // not implemented

#endif  // AETHER_SN_MOMENT_TO_DISCRETE_H_