#ifndef AETHER_PGD_SN_SUBSPACE_EIGEN_H_
#define AETHER_PGD_SN_SUBSPACE_EIGEN_H_

#include <deal.II/lac/block_vector.h>

#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

class SubspaceEigen {
 public:
  double virtual step(
      dealii::BlockVector<double> &modes,
      const std::vector<std::vector<InnerProducts>> &coefficients,
      const double shift) = 0;
  void virtual get_inner_products(
      const dealii::BlockVector<double> &modes,
      std::vector<std::vector<InnerProducts>> &inner_products) = 0;
};

}

#endif  // AETHER_PGD_SN_SUBSPACE_EIGEN_H_