#include "moment_to_discrete.h"

namespace aether::sn {

template <int dim, int qdim>
MomentToDiscrete<dim, qdim>::MomentToDiscrete(
    const QAngle<dim, qdim> &quadrature)
    : quadrature(quadrature) {}

template <int dim, int qdim>
void MomentToDiscrete<dim, qdim>::vmult(dealii::Vector<double> &dst,
                                        const dealii::Vector<double> &src) 
                                        const {
  const int num_ords = quadrature.size();
  const int num_dofs = dst.size() / num_ords;
  dealii::BlockVector<double> dst_b(num_ords, num_dofs);
  dealii::BlockVector<double> src_b(1, num_dofs);
  src_b = src;
  vmult(dst_b, src_b);
  dst = dst_b;
}

template <int dim, int qdim>
template <class Vector>
void MomentToDiscrete<dim, qdim>::vmult(
    dealii::BlockVectorBase<Vector> &dst,
    const dealii::BlockVectorBase<Vector> &src) const {
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int qdim>
void MomentToDiscrete<dim, qdim>::vmult_add(dealii::Vector<double> &dst,
                                            const dealii::Vector<double> &src) 
                                            const {
  const int num_ords = quadrature.size();
  const int num_dofs = dst.size() / num_ords;
  dealii::BlockVector<double> dst_b(num_ords, num_dofs);
  dealii::BlockVector<double> src_b(1, num_dofs);
  dst_b = dst;
  src_b = src;
  vmult_add(dst_b, src_b);
  dst = dst_b;
}

template <int dim, int qdim>
template <class Vector>
void MomentToDiscrete<dim, qdim>::vmult_add(
    dealii::BlockVectorBase<Vector> &dst, 
    const dealii::BlockVectorBase<Vector> &src) const {
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

template class MomentToDiscrete<1>;
template class MomentToDiscrete<2>;
template class MomentToDiscrete<3>;

// Block type is dealii::Vector<double>
template void MomentToDiscrete<1>::vmult<dealii::Vector<double>>(
    dealii::BlockVectorBase<dealii::Vector<double>>&,
    const dealii::BlockVectorBase<dealii::Vector<double>>&) const;
template void MomentToDiscrete<2>::vmult<dealii::Vector<double>>(
    dealii::BlockVectorBase<dealii::Vector<double>>&,
    const dealii::BlockVectorBase<dealii::Vector<double>>&) const;
template void MomentToDiscrete<3>::vmult<dealii::Vector<double>>(
    dealii::BlockVectorBase<dealii::Vector<double>>&,
    const dealii::BlockVectorBase<dealii::Vector<double>>&) const;
// Block type is dealii::PETScWrappers::MPI::Vector
template void MomentToDiscrete<1>::vmult<dealii::PETScWrappers::MPI::Vector>(
    dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&) const;
template void MomentToDiscrete<2>::vmult<dealii::PETScWrappers::MPI::Vector>(
    dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&) const;
template void MomentToDiscrete<3>::vmult<dealii::PETScWrappers::MPI::Vector>(
    dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&) const;

}  // namespace aether::sn