#ifndef AETHER_SN_FISSION_SOURCE_H_
#define AETHER_SN_FISSION_SOURCE_H_

#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_matrix_free.h>
#include <deal.II/lac/petsc_vector_base.h>
#endif

#include "sn/fixed_source.h"
#include "sn/fission_source.h"

namespace aether::sn {

template <int dim, int qdim, class SolverType, class PreconditionType>
class FissionSource {
 public:
  FissionSource(const FixedSource<dim, qdim> &fixed_source,
                const Fission<dim, qdim> &fission,
                SolverType &solver,
                const PreconditionType &preconditioner);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;

 protected:
  const FixedSource<dim, qdim> &fixed_source;
  const Fission<dim, qdim> &fission;
  SolverType &solver;
  const PreconditionType &preconditioner;
};

template <int dim, int qdim, class SolverType, class PreconditionType>
FissionSource<dim, qdim, SolverType, PreconditionType>::
    FissionSource(const FixedSource<dim, qdim> &fixed_source,
                  const Fission<dim, qdim> &fission,
                  SolverType &solver,
                  const PreconditionType &preconditioner)
    : fixed_source(fixed_source), fission(fission), solver(solver), 
      preconditioner(preconditioner) {}

template <int dim, int qdim, class SolverType, class PreconditionType>
void FissionSource<dim, qdim, SolverType, PreconditionType>::
    vmult(dealii::BlockVector<double> &dst, 
          const dealii::BlockVector<double> &src) const {
  dealii::BlockVector<double> fissioned(src.get_block_indices());
  fission.vmult(fissioned, src);
  solver.solve(fixed_source, dst, fissioned, preconditioner);
}

#ifdef DEAL_II_WITH_PETSC
namespace PETScWrappers {

template <int dim, int qdim, class SolverType, class PreconditionType>
class FissionSource : 
    public dealii::PETScWrappers::MatrixFree,
    public ::aether::sn::FissionSource<dim, qdim, SolverType, PreconditionType> {
 public:
  using Base = 
      ::aether::sn::FissionSource<dim, qdim, SolverType, PreconditionType>;
  FissionSource(const FixedSource<dim, qdim> &fixed_source,
                const Fission<dim, qdim> &fission,
                SolverType &solver,
                const PreconditionType &preconditioner);
  using Base::vmult;
  void vmult(dealii::PETScWrappers::VectorBase &dst,
             const dealii::PETScWrappers::VectorBase &src) const;
  void vmult_add(dealii::PETScWrappers::VectorBase &dst,
                 const dealii::PETScWrappers::VectorBase &src) const;
  void Tvmult(dealii::PETScWrappers::VectorBase &dst,
              const dealii::PETScWrappers::VectorBase &src) const;
  void Tvmult_add(dealii::PETScWrappers::VectorBase &dst,
                  const dealii::PETScWrappers::VectorBase &src) const;
};

template <int dim, int qdim, class SolverType, class PreconditionType>
FissionSource<dim, qdim, SolverType, PreconditionType>::
    FissionSource(const FixedSource<dim, qdim> &fixed_source,
                  const Fission<dim, qdim> &fission,
                  SolverType &solver,
                  const PreconditionType &preconditioner)
    : dealii::PETScWrappers::MatrixFree(MPI_COMM_WORLD,
        fixed_source.m(), fixed_source.n(), fixed_source.m(), fixed_source.n()),
      Base(fixed_source, fission, solver, preconditioner) {}

template <int dim, int qdim, class SolverType, class PreconditionType>
void FissionSource<dim, qdim, SolverType, PreconditionType>::
    vmult(dealii::PETScWrappers::VectorBase &dst,
          const dealii::PETScWrappers::VectorBase &src) const {
  const int num_groups = this->fixed_source.within_groups.size();
  const int num_qdofs = dst.size() / num_groups;
  AssertThrow(dst.size() == num_groups * num_qdofs,
              dealii::ExcNotMultiple(dst.size(), num_groups));
  AssertDimension(dst.size(), src.size());
  dealii::BlockVector<double> dst_b(num_groups, num_qdofs);
  dealii::BlockVector<double> src_b(num_groups, num_qdofs);
  for (int g = 0; g < num_groups; ++g) {
    for (int i = 0; i < num_qdofs; ++i) {
      dst_b.block(g)[i] = dst[g*num_qdofs+i];
      src_b.block(g)[i] = src[g*num_qdofs+i];
    }
  }
  vmult(dst_b, src_b);
  for (int g = 0; g < num_groups; ++g)
    for (int i = 0; i < num_qdofs; ++i)
      dst[g*num_qdofs+i] = dst_b.block(g)[i];
}

template <int dim, int qdim, class SolverType, class PreconditionType>
void FissionSource<dim, qdim, SolverType, PreconditionType>::
    vmult_add(dealii::PETScWrappers::VectorBase &dst,
          const dealii::PETScWrappers::VectorBase &src) const {
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, int qdim, class SolverType, class PreconditionType>
void FissionSource<dim, qdim, SolverType, PreconditionType>::
    Tvmult(dealii::PETScWrappers::VectorBase &dst,
          const dealii::PETScWrappers::VectorBase &src) const {
  AssertThrow(false, dealii::ExcNotImplemented());
}

template <int dim, int qdim, class SolverType, class PreconditionType>
void FissionSource<dim, qdim, SolverType, PreconditionType>::
    Tvmult_add(dealii::PETScWrappers::VectorBase &dst,
          const dealii::PETScWrappers::VectorBase &src) const {
  AssertThrow(false, dealii::ExcNotImplemented());
}

}  // namespace PETScWrappers
#endif  // DEAL_WITH_PETSC

}  // namespace aether::sn

#endif  // AETHER_SN_FISSION_SOURCE_H_