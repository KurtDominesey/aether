#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

InnerProducts operator*(const InnerProducts &a, const InnerProducts&b) {
  InnerProducts c(a.scattering.size(), a.scattering[0].size());
  c = 1;
  c *= a;
  c *= b;
  return c;
}

}