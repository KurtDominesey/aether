#ifndef AETHER_PGD_SN_SUBSPACE_JACOBIAN_PC_H_
#define AETHER_PGD_SN_SUBSPACE_JACOBIAN_PC_H_

#include <deal.II/lac/block_vector.h>

#include "pgd/sn/fission_s_problem.h"
#include "pgd/sn/energy_mg_fiss.h"

namespace aether::pgd::sn {

/**
 * Subspace Jacobian preconditioner for the spatio-angular/energy separated
 * eigenvalue problem.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class SubspaceJacobianPC {
 public:
  SubspaceJacobianPC(
      FissionSProblem<dim, qdim> &spatio_angular, 
      EnergyMgFiss &energy,
      const std::vector<std::vector<std::vector<InnerProducts>>> &coefficients,
      const double &k_eigenvalue);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  dealii::BlockVector<double> modes;

 protected:
  FissionSProblem<dim, qdim> &spatio_angular;
  EnergyMgFiss &energy;
  // std::vector<std::vector<pgd::sn::InnerProducts>> &inner_products;
  const std::vector<std::vector<std::vector<pgd::sn::InnerProducts>>> &coefficients;
  const double &k_eigenvalue;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_SUBSPACE_JACOBIAN_PC_H_