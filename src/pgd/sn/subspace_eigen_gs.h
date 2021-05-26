#ifndef AETHER_PGD_SN_SUBSPACE_EIGEN_GS_H_
#define AETHER_PGD_SN_SUBSPACE_EIGEN_GS_H_

#include <deal.II/lac/block_vector.h>

namespace aether::pgd::sn {

class SubspaceEigenGS {
 public:
  SubspaceEigenGS(std::vector<SubspaceEigen> &subspace_ops);
  double step(std::vector<dealii::BlockVector<double>> modes);

 protected:
  std::vector<std::vector<std::vector<InnerProducts>>> inner_products;
  std::vector<std::vector<InnerProducts>> coefficients;
  std::vector<SubspaceEigen> subspace_ops;
}

}

#endif  // AETHER_PGD_SN_SUBSPACE_GS_H_