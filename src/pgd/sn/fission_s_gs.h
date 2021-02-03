#ifndef AETHER_PGD_SN_FISSION_S_GS_H_
#define AETHER_PGD_SN_FISSION_S_GS_H_

#include "pgd/sn/fixed_source_s_gs.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FissionSGS : public FixedSourceSGS<dim, qdim> {
 public:
  using FixedSourceSGS<dim, qdim>::FixedSourceSGS;
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  void set_cross_sections(const std::vector<std::vector<Mgxs>> &mgxs);
  void set_shift(const double shift);

 protected:
  double shift;
  bool shifted = false;
};

}

#endif  // AETHER_PGD_SN_FISSION_S_GS_H_
