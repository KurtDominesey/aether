#include "fixed_source_gs.hpp"

template <int dim, int qdim>
FixedSourceGS<dim, qdim>::FixedSourceGS(
      const std::vector<WithinGroup<dim, qdim>> &within_groups,
      const std::vector<std::vector<ScatteringBlock<dim>>> &downscattering,
      const std::vector<std::vector<ScatteringBlock<dim>>> &upscattering,
      const MomentToDiscrete<qdim> &m2d,
      const DiscreteToMoment<qdim> &d2m)
      : within_groups(within_groups),
        downscattering(downscattering),
        upscattering(upscattering),
        m2d(m2d), d2m(d2m) {}

template <int dim, int qdim>
void FixedSourceGS<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst, 
    const dealii::BlockVector<double> &src) const {
  const int num_groups = within_groups.size();
  const int num_ords = within_groups[0].transport.n_block_cols();
  const int num_dofs = dst.block(0).size() / num_ords;
  AssertDimension(num_groups, dst.n_blocks());
  AssertDimension(num_groups, src.n_blocks());
  AssertDimension(num_groups, upscattering.size());
  AssertDimension(num_groups, downscattering.size());
  dealii::BlockVector<double> dst_m(num_groups, num_dofs);
  dealii::Vector<double> inscattered_m(num_dofs);
  dealii::Vector<double> inscattered(num_ords*num_dofs);
  dealii::SolverControl solver_control(1000, 1e-10);
  dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control);
  for (int g = 0; g < num_groups; ++g) {
    Assert(downscattering[g].size() < g + 1, dealii::ExcInvalidState());
    inscattered_m = 0;
    for (int up = 0; up < downscattering[g].size(); ++up)
      downscattering[g][up].vmult_add(inscattered_m, dst_m.block(g-1-up));
    m2d.vmult(inscattered, inscattered_m);
    solver.solve(within_groups[g], dst.block(g), inscattered,
                 dealii::PreconditionIdentity());
    d2m.vmult(dst_m.block(g), dst.block(g));
  }
}

template class FixedSourceGS<1>;
template class FixedSourceGS<2>;
template class FixedSourceGS<3>;