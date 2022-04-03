#include "mgxs.h"

namespace takeda::lwr {

aether::Mgxs create_mgxs(bool rodded) {
  auto mgxs = aether::Mgxs(2/*groups*/, 3/*materials*/, 1/*legendre moment*/);
  mgxs.total[0] = {2.23775e-1, 2.50367e-1, rodded ? 8.52325e-2 : 1.28407e-2};
  mgxs.total[1] = {1.03864,    1.64482,    rodded ? 2.17460e-1 : 1.20676e-2};
  mgxs.scatter[0][0] = {1.92423e-1, 1.93446e-1, rodded ? 6.77241e-2 : 1.27700e-2};
  mgxs.scatter[0][1] = {0.0, 0.0, 0.0};
  mgxs.scatter[1][0] = {2.28253e-2, 5.65042e-2, rodded ? 6.45461e-5 : 2.40997e-5};
  mgxs.scatter[1][1] = {8.80439e-1, 1.62452,    3.52358e-2, 1.07387e-2};
  mgxs.nu_fission[0] = {9.09319e-3, 0.0, 0.0};
  mgxs.nu_fission[1] = {2.90183e-1, 0.0, 0.0};
  mgxs.chi[0] = {1.0, 0.0, 0.0};
  mgxs.chi[1] = {0.0, 0.0, 0.0};
  return mgxs;
}

}  // namespace takeda::lwr