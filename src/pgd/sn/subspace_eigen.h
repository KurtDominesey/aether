#ifndef AETHER_PGD_SN_SUBSPACE_EIGEN_H_
#define AETHER_PGD_SN_SUBSPACE_EIGEN_H_

#include <deal.II/lac/block_vector.h>

#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

class SubspaceEigen {
 public:
  void virtual residual(
      dealii::Vector<double> &residual,
      const dealii::Vector<double> &modes,
      const double k_eigenvalue,
      const std::vector<std::vector<InnerProducts>> &coefficients) = 0;
//   double virtual step(
//       dealii::BlockVector<double> &modes,
//       const std::vector<std::vector<InnerProducts>> &coefficients,
//       const double shift) = 0;
  void virtual get_inner_products(
      const dealii::Vector<double> &modes,
      std::vector<std::vector<InnerProducts>> &inner_products) = 0;
  double virtual inner_product(const dealii::Vector<double> &left,
                               const dealii::Vector<double> &right) {
    AssertThrow(false, dealii::ExcNotImplemented());
    return 0;
  }
};

}

#endif  // AETHER_PGD_SN_SUBSPACE_EIGEN_H_