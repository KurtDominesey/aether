#ifndef AETHER_PGD_SN_ENERGY_MG_FULL_H_
#define AETHER_PGD_SN_ENERGY_MG_FULL_H_

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/petsc_sparse_matrix.h>

#include "pgd/sn/inner_products.h"
#include "pgd/sn/linear_updatable_interface.h"
#include "base/precondition_growing_lu.h"
#include "base/precondition_block_growing_lu.h"
#include "base/mgxs.h"

namespace aether::pgd::sn {

class EnergyMgFull : public LinearUpdatableInterface {
 public:
  EnergyMgFull(const Mgxs &mgxs,
               const std::vector<dealii::Vector<double>> &sources);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src,
             std::vector<InnerProducts> coefficients_x,
             std::vector<double> coefficients_b);
  double step(dealii::Vector<double> &delta,
            const dealii::BlockVector<double> &b,
            std::vector<InnerProducts> coefficients_x,
            std::vector<double> coefficients_b,
            double omega = 1.0);
  void take_step(const double factor, const dealii::Vector<double> &delta);
  void solve_minimax(const double norm);
  void get_inner_products(std::vector<InnerProducts> &inner_products_x,
                          std::vector<double> &inner_products_b);
  void get_inner_products(std::vector<InnerProducts> &inner_products_x,
                          std::vector<double> &inner_products_b,
                          const int m_row, const int m_col_start);
  void get_inner_products_x(std::vector<InnerProducts> &inner_products_x,
                            const dealii::Vector<double> &left);
  void get_inner_products_x(InnerProducts &inner_products_x,
                            const dealii::Vector<double> &left,
                            const dealii::Vector<double> &right);
  void get_inner_products_x(InnerProducts &inner_products_x,
                            const dealii::Vector<double> &right);
  void get_inner_products_b(std::vector<double> &inner_products_b,
                            const dealii::Vector<double> &left);
  double line_search(
      std::vector<double> &c,
      const dealii::Vector<double> &step,
      const InnerProducts &coefficients_x_mode_step,
      const InnerProducts &coefficients_x_step_step,
      std::vector<InnerProducts> coefficients_x_mode_mode,
      std::vector<InnerProducts> coefficients_x_step_mode,
      const std::vector<double> &coefficients_b_mode,
      const std::vector<double> &coefficients_b_step);
  double enrich(const double factor);
  double normalize();
  void scale(double factor);
  double get_residual(std::vector<InnerProducts> coefficients_x,
                      std::vector<double> coefficients_b);
  double get_norm() const;
  void update(std::vector<std::vector<InnerProducts>> coefficients_x,
              std::vector<std::vector<double>> coefficients_b);
  std::vector<dealii::Vector<double>> modes;
  std::vector<dealii::Vector<double>> modes_adj;
  const std::vector<dealii::Vector<double>> &sources;
  const Mgxs &mgxs;
 protected:
  dealii::FullMatrix<double> matrix;
  dealii::Vector<double> source;
  std::vector<dealii::Vector<double>> test_funcs;
  PreconditionBlockGrowingLU<dealii::PETScWrappers::SparseMatrix, double> gs_lu;
  virtual void set_matrix(InnerProducts coefficients_x);
  virtual void set_source(std::vector<double> coefficients_b, 
                          std::vector<InnerProducts> coefficients_x);
};

}

#endif  // AETHER_PGD_SN_ENERGY_MG_FULL_H_