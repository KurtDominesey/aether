#ifndef AETHER_PGD_SN_EIGEN_UPDATABLE_INTERFACE_H_
#define AETHER_PGD_SN_EIGEN_UPDATABLE_INTERFACE_H_

#include "pgd/sn/eigen_interface.h"

namespace aether::pgd::sn {

class EigenUpdatableInterface : public EigenInterface {
 public:
  virtual double update(std::vector<std::vector<InnerProducts>> &coefficients,
                        const double tol=1e-5,
                        const std::string eps_type="krylovschur",
                        const int num_modes=-1) = 0;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_EIGEN_UPDATABLE_INTERFACE_H_