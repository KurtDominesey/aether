#include "pgd/sn/subspace_jacobian_fd.h"

namespace aether::pgd::sn {

SubspaceJacobianFD::SubspaceJacobianFD(std::vector<SubspaceEigen*> ops, 
                                       const int num_modes,
                                       const int num_materials,
                                       const int num_legendre)
    : ops(ops), num_modes(num_modes) {
  coefficients.resize(num_modes, std::vector<InnerProducts>(
      num_modes, InnerProducts(num_materials, num_legendre)));
  inner_products.resize(ops.size(), coefficients);
  inner_products_unperturbed = inner_products;
}

void SubspaceJacobianFD::set_modes(const dealii::BlockVector<double> &modes) {
  unperturbed = modes;
  residual_unperturbed.reinit(modes);
  residual(residual_unperturbed, modes);
  residual_unperturbed *= -1;
  inner_products_unperturbed.clear();
  inner_products_unperturbed = inner_products;
  k_eigenvalue = modes.block(modes.n_blocks()-1)[0];
  std::cout << "residual norm: " << residual_unperturbed.l2_norm() << " ***\n";
  std::cout << "k-eigenvalue: " << modes.block(modes.n_blocks()-1)[0] << "\n";
}

void SubspaceJacobianFD::residual(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &modes) const {
  for (int i = 0; i < ops.size(); ++i) {
    ops[i]->get_inner_products(modes.block(i), inner_products[i]);
  }
  const double k_eigenvalue = modes.block(modes.n_blocks()-1)[0];
  double norm = 0;
  for (int i = 0; i < ops.size(); ++i) {
    for (int m = 0; m < num_modes; ++m) { 
      for (int mp = 0; mp < num_modes; ++mp) {
        coefficients[m][mp] = 1;
        for (int j = 0; j < ops.size(); ++j) {
          if (i != j) {
            coefficients[m][mp] *= inner_products[j][m][mp];
          }
        }
      }
    }
    ops[i]->residual(dst.block(i), modes.block(i), k_eigenvalue, coefficients);
    if (i == modes.n_blocks()-2)
      norm += std::pow(modes.block(i).l2_norm(), 2);
      // for (int g = 0; g < modes.block(i).size(); ++g)
      //   norm += modes.block(i)[g];
      // norm += ops[i]->inner_product(modes.block(i), modes.block(i));
  }
  dst.block(modes.n_blocks()-1)[0] = 1e2 * (norm - 1);
}


void SubspaceJacobianFD::vmult(dealii::BlockVector<double> &dst,
                               const dealii::BlockVector<double> &src) const {
  const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
  // const double epsilon = 1e-6;
  // std::cout << "epsilon is " << epsilon << "\n";
  double scale = (epsilon * unperturbed.l1_norm()) / 
                 (src.size() * src.l2_norm()) + epsilon;
  // scale = 1e-4;
  dealii::BlockVector<double> perturbed(unperturbed);
  perturbed.add(scale, src);
  dst = 0;
  residual(dst, perturbed);
  double perturbed_norm = dst.l2_norm();
  dst += residual_unperturbed;  // += -F(u)
  dst /= scale;
}

}  // namespace aether::pgd::sn