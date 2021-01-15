#ifndef AETHER_PGD_SN_EIGEN_GS_H_
#define AETHER_PGD_SN_EIGEN_GS_H_

#include "pgd/sn/nonlinear_gs.h"
#include "pgd/sn/eigen_interface.h"
#include "pgd/sn/eigen_updatable_interface.h"

namespace aether::pgd::sn {

class EigenGS : public NonlinearGS {
 public:
  EigenGS(std::vector<LinearInterface*>& linear_ops,
          int num_materials, int num_legendre);
  double initialize_iteratively(double tol);
  double initialize_guess();
  double update() override;
 protected:
  std::vector<EigenInterface*> eigen_ops;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_EIGEN_GS_H_