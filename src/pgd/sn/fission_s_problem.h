#ifndef AETHER_PGD_SN_FISSION_S_PROBLEM_H_
#define AETHER_PGD_SN_FISSION_S_PROBLEM_H_

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/slepc_solver.h>

#include "base/petsc_block_wrapper.h"
#include "base/petsc_precondition_shell.h"
#include "base/slepc_transformation_preconditioner.h"

#include "pgd/sn/fixed_source_s_problem.h"
#include "pgd/sn/fission_s.h"
#include "pgd/sn/fission_s_gs.h"
#include "pgd/sn/subspace_eigen.h"
#include "pgd/sn/shifted_s.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FissionSProblem : public FixedSourceSProblem<dim, qdim>, 
                        public SubspaceEigen {
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

  /**
   * Update the modes.
   */
  double step(dealii::BlockVector<double> &modes,
              const std::vector<std::vector<InnerProducts>> &coefficients,
              const double shift);

  /**
   * Compute inner products.
   */
  void get_inner_products(
      const dealii::BlockVector<double> &modes,
      std::vector<std::vector<InnerProducts>> &inner_products);

 protected:
  /**
   * Update the modes using Generalized Davidson after setting coefficients.
   */
  double step_gd(dealii::BlockVector<double> &modes, const double shift);

  /**
   * Update the modes using Shifted Power after setting coefficients.
   */
  double step_power_shift(dealii::BlockVector<double> &modes, 
                          const double shift);

  std::vector<std::vector<aether::sn::Emission<dim>>> emission;
  std::vector<std::vector<aether::sn::Production<dim>>> production;

 public:
  FissionS<dim, qdim> fission_s;
  FissionSGS<dim, qdim> fission_s_gs;
};

}

#endif  // AETHER_PGD_SN_FISSION_S_PROBLEM_H_