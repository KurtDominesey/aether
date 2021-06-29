#ifndef AETHER_PGD_SN_ENERGY_MG_FISS_H_
#define AETHER_PGD_SN_ENERGY_MG_FISS_H_

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>

#include "base/petsc_wrapper.h"
#include "base/petsc_precondition_shell.h"
#include "base/slepc_transformation_preconditioner.h"
#include "base/petsc_solver.h"
#include "pgd/sn/energy_mg_full.h"
#include "pgd/sn/eigen_updatable_interface.h"
#include "pgd/sn/subspace_eigen.h"

namespace aether::pgd::sn {

class EnergyMgFiss : public EnergyMgFull, public EigenUpdatableInterface,
                     public SubspaceEigen {
 public:
  const std::vector<dealii::Vector<double>> zero_sources;
  EnergyMgFiss(const Mgxs &mgxs);
  double step_eigenvalue(InnerProducts &coefficients);
  double update(std::vector<std::vector<InnerProducts>> &coefficients,
                const double tol=1e-6, const std::string eps_type="krylovschur",
                int num_modes=-1);
  void update(std::vector<std::vector<InnerProducts>> coefficients_x,
              std::vector<std::vector<double>> coefficients_b) override;
  void  residual(dealii::Vector<double> &residual,
                 const dealii::Vector<double> &modes,
                 const double k_eigenvalue,
                 const std::vector<std::vector<InnerProducts>> &coefficients) 
                 override;
  double inner_product(const dealii::Vector<double> &left, 
                       const dealii::Vector<double> &right) override;
  using EnergyMgFull::get_inner_products;
  void get_inner_products(
      const dealii::Vector<double> &modes,
      std::vector<std::vector<InnerProducts>> &inner_products) override;
  void solve_fixed_k(
      dealii::Vector<double> &dst,
      const dealii::Vector<double> &src,
      const double k_eigenvalue,
      const std::vector<std::vector<InnerProducts>> &coefficients);

 protected:
  void set_matrix(InnerProducts coefficients_x) override;
  void set_source(std::vector<double> coefficients_b, 
                  std::vector<InnerProducts> coefficients_x) override;
  void set_fixed_k_matrix(
      dealii::FullMatrix<double> &a_minus_kb,
      const double k_eigenvalue,
      const std::vector<std::vector<InnerProducts>> &coefficients);
  std::unique_ptr<dealii::PETScWrappers::PreconditionerBase> preconditioner_ptr;
  std::unique_ptr<dealii::PETScWrappers::PreconditionerBase> preconditioner_adj_ptr;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_ENERGY_MG_FISS_H_