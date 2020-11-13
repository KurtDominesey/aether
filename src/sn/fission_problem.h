#ifndef AETHER_SN_FISSION_PROBLEM_H_
#define AETHER_SN_FISSION_PROBLEM_H_

#include "sn/fixed_source_problem.h"
#include "sn/fission.h"
#include "sn/emission.h"
#include "sn/production.h"

namespace aether::sn {

/**
 * Convenience class which sets up a fission (k-eigenvlaue) problem.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2,
          class TransportType = Transport<dim, qdim>,
          class TransportBlockType = TransportBlock<dim, qdim>>
class FissionProblem : 
    public FixedSourceProblem<dim, qdim, TransportType, TransportBlockType> {
 public:
  FissionProblem(const dealii::DoFHandler<dim> &dof_handler,
                 const QAngle<dim, qdim> &quadrature,
                 const Mgxs &mgxs,
                 const std::vector<std::vector<dealii::BlockVector<double>>>
                    &boundary_conditions);
  //! Fission operator
  Fission<dim, qdim> fission;
  //! Emission operator
  Emission<dim> emission;
  //! Production operator
  Production<dim> production;
};

template <int dim, int qdim, class TransportType, class TransportBlockType>
FissionProblem<dim, qdim, TransportType, TransportBlockType>::
    FissionProblem(
        const dealii::DoFHandler<dim> &dof_handler,
        const QAngle<dim, qdim> &quadrature,
        const Mgxs &mgxs,
        const std::vector<std::vector<dealii::BlockVector<double>>> 
            &boundary_conditions)
    : FixedSourceProblem<dim, qdim, TransportType, TransportBlockType>(
          dof_handler, quadrature, mgxs, boundary_conditions),
      emission(dof_handler, mgxs.chi),
      production(dof_handler, mgxs.nu_fission),
      fission(this->within_groups, this->m2d, emission, production, this->d2m)
      {}

}  // namespace aether::sn

#endif  // AETHER_SN_FISSION_PROBLEM_H_