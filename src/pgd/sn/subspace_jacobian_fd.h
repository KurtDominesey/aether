#ifndef AETHER_PGD_SN_SUBSPACE_JACOBIAN_FD_H_
#define AETHER_PGD_SN_SUBSPACE_JACOBIAN_FD_H_

#include <deal.II/lac/block_vector.h>

#include "base/block_block_vector.h"
#include "pgd/sn/subspace_eigen.h"
#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

/**
 * Finite difference (FD) approximation of the PGD subspace Jacobian.
 * 
 * Used in Jacobian Free Newton Krylov (JFNK) methods.
 */
class SubspaceJacobianFD {
 public:
  SubspaceJacobianFD(std::vector<SubspaceEigen*> ops, const int num_modes, 
                     const int num_materials, const int num_legendre);
  void set_modes(const dealii::BlockVector<double> &modes);
  void residual(dealii::BlockVector<double> &dst,
                const dealii::BlockVector<double> &modes) const;
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  dealii::BlockVector<double> residual_unperturbed;  // -F(u)
  std::vector<std::vector<std::vector<InnerProducts>>> inner_products_unperturbed;
  double k_eigenvalue = 1;

 protected:
  mutable std::vector<SubspaceEigen*> ops;
  dealii::BlockVector<double> unperturbed;  // u
  mutable std::vector<std::vector<std::vector<InnerProducts>>> inner_products;
  mutable std::vector<std::vector<InnerProducts>> coefficients;
  const int num_modes;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_SUBSPACE_JACOBIAN_FD_H_