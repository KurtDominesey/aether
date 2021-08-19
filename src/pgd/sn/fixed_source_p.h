#ifndef AETHER_PGD_SN_FIXED_SOURCE_P_H_
#define AETHER_PGD_SN_FIXED_SOURCE_P_H_

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>

#include "base/mgxs.h"
#include "sn/fixed_source.h"
#include "pgd/sn/transport_block.h"
#include "pgd/sn/inner_products.h"
#include "pgd/sn/linear_interface.h"

namespace aether::pgd::sn {

struct Cache {
  Cache(int num_groups, int num_ordinates, int num_moments, int num_spatial)
      : mode(num_groups, num_ordinates * num_spatial),
        moments(num_groups, num_moments * num_spatial),
        streamed(num_groups, num_ordinates * num_spatial) {};
  dealii::BlockVector<double> mode;
  dealii::BlockVector<double> moments;
  dealii::BlockVector<double> streamed;
};

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSourceP : public LinearInterface {
 public:
  FixedSourceP(aether::sn::FixedSource<dim, qdim> &fixed_source,
               Mgxs &mgxs_psuedo, const Mgxs &mgxs,
               std::vector<dealii::BlockVector<double>> &sources);
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
  void get_inner_products_x(std::vector<InnerProducts> &inner_products,
                            const dealii::Vector<double> &left);
  void get_inner_products_x(InnerProducts &inner_products,
                            const dealii::Vector<double> &left,
                            const dealii::Vector<double> &right);
  void get_inner_products_x(InnerProducts &inner_products,
                            const dealii::Vector<double> &right);
  void get_inner_products_x(InnerProducts &inner_products,
                            const dealii::BlockVector<double> &left,
                            const Cache &right);
  void get_inner_products_b(std::vector<double> &inner_products,
                            const dealii::Vector<double> &left);
  void get_inner_products_b(std::vector<double> &inner_products, 
                            const dealii::BlockVector<double> &left);
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
  std::vector<Cache> caches;
  const std::vector<dealii::BlockVector<double>> &sources;
  aether::sn::FixedSource<dim, qdim> &fixed_source;
  void set_cache(Cache &cache);
 protected:
  const Mgxs &mgxs;
  Mgxs &mgxs_pseudo;
  std::vector<dealii::BlockVector<double>> test_funcs;
  void set_last_cache();
  virtual void set_cross_sections(const InnerProducts &coefficients_x);
  void get_source(dealii::BlockVector<double> &source,
                  const std::vector<InnerProducts> &coefficients_x,
                  const std::vector<double> &coefficients_b,
                  double denominator);
  virtual void subtract_modes_from_source(
      dealii::BlockVector<double> &source,
      std::vector<InnerProducts> coefficients_x);
  void get_inner_products_x(std::vector<InnerProducts> &inner_products,
                            const int m_row, const int m_col_start);
  void get_inner_products_b(std::vector<double> &inner_products, 
                            const int m_row);
};


}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_P_H_

