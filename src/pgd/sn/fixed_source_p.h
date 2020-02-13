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
  void step(dealii::BlockVector<double> &x,
            const dealii::BlockVector<double> &b,
            std::vector<InnerProducts> coefficients_x,
            std::vector<double> coefficients_b);
  void get_inner_products(std::vector<InnerProducts> &inner_products_x,
                          std::vector<double> &inner_products_b);
  void enrich();
  void normalize();
  std::vector<Cache> caches;
 protected:
  aether::sn::FixedSource<dim, qdim> &fixed_source;
  const Mgxs &mgxs;
  Mgxs &mgxs_pseudo;
  const std::vector<dealii::BlockVector<double>> &sources;
  void set_last_cache();
  void set_cross_sections(InnerProducts &coefficients_x);
  void get_source(dealii::BlockVector<double> &source,
                  std::vector<InnerProducts> &coefficients_x,
                  std::vector<double> &coefficients_b,
                  double denominator);
  void subtract_modes_from_source(dealii::BlockVector<double> &source,
                                  std::vector<InnerProducts> coefficients_x);
  void get_inner_products_x(std::vector<InnerProducts> &inner_products);
  void get_inner_products_b(std::vector<double> &inner_products);
};


}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_P_H_

