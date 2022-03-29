#ifndef AETHER_PGD_SN_NONLINEAR_2D1D_H_
#define AETHER_PGD_SN_NONLINEAR_2D1D_H_

#include "pgd/sn/inner_products.h"
#include "pgd/sn/fixed_source_2D1D.h"

namespace aether::pgd::sn {

template <int zones2D, int zones1D, int groups2D, int groups1D>
class Nonlinear2D1D {
 public:
  using OneD = FixedSource2D1D<1, 1, zones1D, groups1D>;
  using TwoD = FixedSource2D1D<2, 2, zones2D, groups2D>;
  Nonlinear2D1D(OneD &oneD, TwoD &twoD, const Mgxs &mgxs);
  void enrich();
  double iter();
 protected:
  OneD &oneD;
  TwoD &twoD;
  const Mgxs &mgxs;
  std::array<std::array<int, zones2D>, zones1D> materials;
};

template <int zones2D, int zones1D, int groups2D, int groups1D>
Nonlinear2D1D<zones2D, zones1D, groups2D, groups1D>::Nonlinear2D1D(
    OneD &oneD, TwoD &twoD, const Mgxs &mgxs) 
    : oneD(oneD), twoD(twoD), mgxs(mgxs) {}

template <int zones2D, int zones1D, int groups2D, int groups1D>
void Nonlinear2D1D<zones2D, zones1D, groups2D, groups1D>::enrich() {
  oneD.enrich();
  twoD.enrich();
}

template <int zones2D, int zones1D, int groups2D, int groups1D>
double Nonlinear2D1D<zones2D, zones1D, groups2D, groups1D>::iter() {
  oneD.setup<zones2D, groups2D, zones1D, zones2D>(
      twoD.iprods_flux, twoD.iprods_src, materials, mgxs);
  twoD.setup<zones1D, groups1D, zones1D, zones2D>(
      oneD.iprods_flux, oneD.iprods_src, materials, mgxs);
  return 0;
}

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_NONLINEAR_2D1D_H_