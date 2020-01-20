#ifndef AETHER_PGD_SN_FIXED_SOURCE_H_
#define AETHER_PGD_SN_FIXED_SOURCE_H_

#include <deal.II/base/table.h>

#include "sn/fixed_source.h"
#include "pgd/sn/transport.h"

namespace aether::pgd::sn {

struct InnerProducts {
  double streaming;
  dealii::Table<1, double> collision;
  dealii::Table<2, double> scattering;
  dealii::Table<1, double> source;
};

struct Cache {
  std::vector<dealii::BlockVector<double>> modes;
  std::vector<dealii::BlockVector<double>> streamed;
  std::vector<dealii::BlockVector<double>> collided;
  std::vector<dealii::BlockVector<double>> scattered;
};


template <int dim, int qdim>
class FixedSource : public aether::sn::FixedSource<dim, qdim> {
 public:
  void update_cache(const dealii::BlockVector<double> &mode);
  void update_inner_products(const dealii::BlockVector<double> &mode);
  Cache cache;
  InnerProducts inner_products;
 protected:
  void stream(dealii::BlockVector<double> &dst, 
              const dealii::BlockVector<double> &src) const;
  void collide(dealii::BlockVector<double> &dst,
               const dealii::BlockVector<double> &src) const;
  void scatter(dealii::BlockVector<double> &dst,
               const dealii::BlockVector<double> &src) const;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_H_