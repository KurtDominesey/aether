#ifndef AETHER_PGD_SN_ENERGY_MG_FISS_H_
#define AETHER_PGD_SN_ENERGY_MG_FISS_H_

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/petsc_solver.h>

#include "base/petsc_wrapper.h"
#include "base/petsc_precondition_shell.h"
#include "base/slepc_transformation_preconditioner.h"
#include "pgd/sn/energy_mg_full.h"
#include "pgd/sn/eigen_updatable_interface.h"

namespace aether::pgd::sn {

class EnergyMgFiss : public EnergyMgFull, public EigenUpdatableInterface {
 public:
  const std::vector<dealii::Vector<double>> zero_sources;
  EnergyMgFiss(const Mgxs &mgxs);
  double step_eigenvalue(InnerProducts &coefficients);
  double update(std::vector<std::vector<InnerProducts>> &coefficients,
                const double tol=1e-5);
  void update(std::vector<std::vector<InnerProducts>> coefficients_x,
              std::vector<std::vector<double>> coefficients_b) override;
  void set_matrix(InnerProducts coefficients_x) override;
  void set_source(std::vector<double> coefficients_b, 
                  std::vector<InnerProducts> coefficients_x) override;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_ENERGY_MG_FISS_H_