#ifndef AETHER_PGD_NONLINEAR_H_
#define AETHER_PGD_NONLINEAR_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include "pgd/sn/linear_interface.h"
#include "pgd/sn/linear_updatable_interface.h"
#include "pgd/sn/inner_products.h"
#include "pgd/sn/energy_mg_full.h"
#include "pgd/sn/fixed_source_p.h"

#include "sn/moment_to_discrete.h"

namespace aether::pgd::sn {

class NonlinearGS {
 public:
  NonlinearGS(std::vector<LinearInterface*> &linear_ops, 
              int num_materials, int num_legendre, int num_sources);
  double step(dealii::BlockVector<double> x, 
              const dealii::BlockVector<double> b,
              const bool should_normalize = true,
              const bool should_line_search = false);
  virtual double update();
  void reweight();
  void vmult(dealii::BlockVector<double> dst,
             const dealii::BlockVector<double> src);
  void enrich();
  void set_inner_products();
  void finalize();
  void unfinalize();
  std::vector<std::vector<InnerProducts>> inner_products_x;
  std::vector<std::vector<double>> inner_products_b;
  std::vector<std::vector<std::vector<InnerProducts>>> inner_products_all_x;
  std::vector<std::vector<std::vector<double>>> inner_products_all_b;
 protected:
  double get_residual() const;
  void set_coefficients(int i, std::vector<InnerProducts> &coefficients_x, 
                        std::vector<double> &coefficients_b) const;
  std::vector<LinearInterface*> &linear_ops;
  InnerProducts inner_products_one;
  double line_search(const std::vector<dealii::Vector<double>> &steps);
  void expand_mode(dealii::BlockVector<double> &mode,
                   const pgd::sn::Cache &cache_spaceangle, 
                   const dealii::Vector<double> &mode_energy);
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_NONLINEAR_H_