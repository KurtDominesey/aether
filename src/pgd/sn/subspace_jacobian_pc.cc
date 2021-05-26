#include "pgd/sn/subspace_jacobian_pc.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
SubspaceJacobianPC<dim, qdim>::SubspaceJacobianPC(
    FissionSProblem<dim, qdim> &spatio_angular, EnergyMgFiss &energy,
    const std::vector<std::vector<std::vector<InnerProducts>>> &coefficients,
    const double &k_eigenvalue)
    : spatio_angular(spatio_angular), energy(energy), 
      coefficients(coefficients), k_eigenvalue(k_eigenvalue) {}

template <int dim, int qdim>
void SubspaceJacobianPC<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src) const {
  dst = src;
  /*
  spatio_angular.set_cross_sections(coefficients[1]);
  dst.block(0) = 1;
  // spatio_angular.fixed_source_s_gs.vmult(dst.block(0), src.block(0));
  // dst.block(0) /= -k_eigenvalue;
  if (!spatio_angular.fission_s_gs.shifted)
    spatio_angular.fission_s_gs.set_shift(k_eigenvalue);
  spatio_angular.fission_s_gs.vmult(dst.block(0), src.block(0));
  // dealii::ReductionControl control(20, 1e-6, 1e-2);
  // dealii::SolverGMRES<dealii::Vector<double> solver(control);
  */
  dealii::Vector<double> dst_aug(dst.block(1).size()+1);
  dealii::Vector<double> src_aug(src.block(1).size()+1);
  for (int i = 0; i < dst.block(1).size(); ++i) {
    dst_aug[i] = dst.block(1)[i];
    src_aug[i] = src.block(1)[i];
  }
  energy.solve_fixed_k(dst_aug, src_aug, k_eigenvalue, 
                       coefficients[0]);
  for (int i = 0; i < dst.block(1).size(); ++i)
    dst.block(1)[i] = dst_aug[i];
  dst.block(2)[0] = dst_aug[dst_aug.size()-1];
  // return;
  // dst.block(2)[0] = -2*(modes.block(1)*dst.block(1));
  // std::cout << "delta-k: " << dst.block(2)[0] << "\n";
  dealii::Vector<double> src0(src.block(0));
  dealii::Vector<double> bx(src0.size());
  spatio_angular.fixed_source_s.vmult(bx, modes.block(0));
  src0.add(dst.block(2)[0], bx);
  spatio_angular.solve_fixed_k(dst.block(0), src0, k_eigenvalue, 
                               coefficients[1]);
  return;
  /*
  // const double k_eigenvalue = 
  // energy.solve_fixed_k(dst.block(1), src.block(1), coefficients[1])
  const int num_modes = energy.modes.size();
  const int num_groups = energy.modes[0].size();
  dealii::BlockVector<double> modes(num_modes, spatio_angular.transport.m());
  modes = src.block(0);
  for (int m = 0; m < num_modes; ++m)
    for (int g = 0; g < num_groups; ++g)
      energy.modes[m][g] = src.block(1)[m*num_groups+g];
  // solve energy modes
  // for (int m = 0; m < num_modes; ++m)
  //   for (int mp = 0; mp < num_modes; ++mp)
  //     inner_products[m][mp] = 0;
  // spatio_angular.fixed_source_s.get_inner_products_lhs(
  //     inner_products, modes);
  double k_ = energy.update(coefficients[1], 1e-5);
  for (int m = 0; m < num_modes; ++m)
    for (int g = 0; g < num_groups; ++g)
       dst.block(1)[m*num_groups+g] = energy.modes[m][g];
  std::cout << k_eigenvalue << "\n";
  double k_eigenvalue_old = dst.block(2)[0];
  dst.block(2)[0] = k_eigenvalue;
  // solve spatio-angular modes
  dst.block(0) = src.block(0);
  // for (int m = 0; m < num_modes; ++m)
  //   for (int mp = 0; mp < num_modes; ++mp)
  //     inner_products[m][mp] = 0;
  // std::vector<double> _;
  // for (int m = 0; m < num_modes; ++m)
  //     energy.get_inner_products(inner_products[m], _, m, 0);
  // double residual = spatio_angular.step(modes, coefficients[0], 0);
  // dst.block(0) = modes;
  // dst.block(0) /= -k_eigenvalue_old;
  */
}

template class SubspaceJacobianPC<1>;
template class SubspaceJacobianPC<2>;
template class SubspaceJacobianPC<3>;

}  // namespace aether::pgd::sn