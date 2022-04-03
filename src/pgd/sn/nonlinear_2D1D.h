#ifndef AETHER_PGD_SN_NONLINEAR_2D1D_H_
#define AETHER_PGD_SN_NONLINEAR_2D1D_H_

#include "pgd/sn/inner_products.h"
#include "pgd/sn/fixed_source_2D1D.h"

namespace aether::pgd::sn {

class Nonlinear2D1D {
 public:
  Nonlinear2D1D(FixedSource2D1D<1> &one_d, FixedSource2D1D<2> &two_d, 
                const std::vector<std::vector<int>> &materials,
                const Mgxs &mgxs);
  void enrich();
  double iter();
 protected:
  FixedSource2D1D<1> &one_d;
  FixedSource2D1D<2> &two_d;
  const Mgxs &mgxs;
  const std::vector<std::vector<int>> &materials;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_NONLINEAR_2D1D_H_