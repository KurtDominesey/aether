#ifndef AETHER_PGD_SN_EIGEN_INTERFACE_H_
#define AETHER_PGD_SN_EIGEN_INTERFACE_H_

#include "pgd/sn/linear_interface.h"

namespace aether::pgd::sn {

class EigenInterface {
 public:
  virtual double step_eigenvalue(InnerProducts &coefficients) = 0;
  double eigenvalue = 1;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_EIGEN_INTERFACE_H_