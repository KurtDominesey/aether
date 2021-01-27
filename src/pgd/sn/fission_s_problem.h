#ifndef AETHER_PGD_SN_FISSION_S_PROBLEM_H_
#define AETHER_PGD_SN_FISSION_S_PROBLEM_H_

#include "pgd/sn/fixed_source_s_problem.h"
#include "pgd/sn/fission_s.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FissionSProblem : public FixedSourceSProblem<dim, qdim> {
 public:
  /**
   * Constructor.
   */
  FissionSProblem(
      const dealii::DoFHandler<dim> &dof_handler,
      const aether::sn::QAngle<dim, qdim> &quadrature,
      const Mgxs &mgxs,
      const std::vector<std::vector<dealii::BlockVector<double>>>
        &boundary_conditions,
      const int num_modes);

  /**
   * Set cross-sections.
   */
  void set_cross_sections(
      const std::vector<std::vector<InnerProducts>> &coefficients);

 protected:
  std::vector<std::vector<aether::sn::Emission<dim>>> emission;
  std::vector<std::vector<aether::sn::Production<dim>>> production;

 public:
  FissionS<dim, qdim> fission_s;
};

}

#endif  // AETHER_PGD_SN_FISSION_S_PROBLEM_H_