#ifndef AETHER_PGD_SN_FIXED_SOURCE_P_H_
#define AETHER_PGD_SN_FIXED_SOURCE_P_H_

#include <deal.II/base/table.h>

#include "sn/fixed_source.h"
#include "pgd/sn/transport.h"

namespace aether::pgd::sn {

struct InnerProducts {
  InnerProducts(int num_materials, int num_legendre)
      : collision(num_materials),
        scattering(num_materials, num_legendre) {};
  double streaming;
  dealii::Table<1, double> collision;
  dealii::Table<2, double> scattering;
};

struct Cache {
  Cache(int num_ordinates, int num_moments, int num_spatial)
      : mode(num_ordinates, num_spatial),
        moments(num_moments, num_spatial),
        streamed(num_ordinates, num_spatial) {};
  dealii::BlockVector<double> mode;
  dealii::BlockVector<double> moments;
  dealii::BlockVector<double> streamed;
};

struct CrossSections {
  std::vector<double> total;
  std::vector<std::vector<double>> scattering;
};

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSourceP {
 public:
  FixedSourceP(Transport<dim, qdim> &transport,
               aether::sn::Scattering<dim> &scattering,
               aether::sn::FixedSource<dim, qdim> &fixed_source,
               std::vector<double> &xs_total,
               std::vector<std::vector<double>> &xs_scatter);
  void vmult(dealii::BlockVector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  void update_last_cache(const dealii::BlockVector<double> &mode);
  void get_inner_products(const dealii::BlockVector<double> &mode);
  std::vector<Cache> caches;
  std::vector<double> xs_total;
  std::vector<std::vector<double>> xs_scatter;
 protected:
  Transport<dim, qdim> &transport;
  aether::sn::Scattering<dim> &scattering;
  aether::sn::FixedSource<dim, qdim> &fixed_source;
  void set_cross_sections(InnerProducts &inner_products);
  void set_right_hand_side(std::vector<InnerProducts> &inner_products);
};


}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_P_H_

