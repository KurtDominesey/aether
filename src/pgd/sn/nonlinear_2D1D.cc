#include "pgd/sn/nonlinear_2D1D.h"

namespace aether::pgd::sn {

Nonlinear2D1D::Nonlinear2D1D(
    FixedSource2D1D<1> &one_d, FixedSource2D1D<2> &two_d, 
    const std::vector<std::vector<int>> &materials, const Mgxs &mgxs) 
    : one_d(one_d), two_d(two_d), materials(materials), mgxs(mgxs) {}

void Nonlinear2D1D::enrich() {
  one_d.enrich();
  two_d.enrich();
  // one_d.normalize();
  two_d.normalize();
  one_d.set_inner_prods();
  two_d.set_inner_prods();
}

double Nonlinear2D1D::iter() {
  one_d.normalize();
  one_d.set_inner_prods();
  two_d.setup(one_d.iprods_flux, one_d.iprods_src, materials, mgxs);
  double r2 = two_d.solve();
  two_d.normalize();
  two_d.set_inner_prods();
  one_d.setup(two_d.iprods_flux, two_d.iprods_src, materials, mgxs);
  double r1 = one_d.solve();
  double r = std::sqrt(r2*r2+r1*r1);
  std::cout << "2D: " << r2 << "  "
            << "1D: " << r1 << "  "
            << "2D/1D: " << r << "\n";
  return 0;
}

}