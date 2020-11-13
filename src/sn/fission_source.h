#ifndef AETHER_SN_FISSION_SOURCE_H_
#define AETHER_SN_FISSION_SOURCE_H_

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

}

#endif  // AETHER_SN_FISSION_SOURCE_H_