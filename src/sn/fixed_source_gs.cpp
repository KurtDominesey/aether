#include "fixed_source_gs.hpp"

namespace aether::sn {

template <class SolverType, int dim, int qdim>
FixedSourceGS<SolverType, dim, qdim>::FixedSourceGS(
      const std::vector<WithinGroup<dim, qdim>> &within_groups,
      const std::vector<std::vector<ScatteringBlock<dim>>> &downscattering,
      const std::vector<std::vector<ScatteringBlock<dim>>> &upscattering,
      const MomentToDiscrete<qdim> &m2d,
      const DiscreteToMoment<qdim> &d2m,
      SolverType &solver)
      : within_groups(within_groups),
        downscattering(downscattering),
        upscattering(upscattering),
        m2d(m2d), 
        d2m(d2m),
        solver(solver) {}

template <class SolverType, int dim, int qdim>
void FixedSourceGS<SolverType, dim, qdim>::vmult(
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
  dealii::Vector<double> downscattered_m(num_dofs);
  dealii::Vector<double> downscattered(num_ords*num_dofs);
  dealii::Vector<double> transported(num_ords*num_dofs);
  for (int g = 0; g < num_groups; ++g) {
    Assert(downscattering[g].size() < g + 1, dealii::ExcInvalidState());
    transported = 0;
    downscattered_m = 0;
    for (int up = 0; up < downscattering[g].size(); ++up)
      downscattering[g][up].vmult_add(downscattered_m, dst_m.block(g-1-up));
    m2d.vmult(downscattered, downscattered_m);
    within_groups[g].transport.vmult(transported, downscattered);
    transported += src.block(g);
    solver.solve(within_groups[g], dst.block(g), transported,
                 dealii::PreconditionIdentity());
    d2m.vmult(dst_m.block(g), dst.block(g));
  }
}

template <class SolverType, int dim, int qdim>
void FixedSourceGS<SolverType, dim, qdim>::step(
    dealii::BlockVector<double> &flux,
    const dealii::BlockVector<double> &src) const {
  const int num_groups = within_groups.size();
  const int num_ords = within_groups[0].transport.n_block_cols();
  const int num_dofs = flux.block(0).size() / num_ords;
  dealii::BlockVector<double> flux_m(num_groups, num_dofs);
  dealii::BlockVector<double> upscattered_m(num_groups, num_dofs);
  dealii::BlockVector<double> upscattered(num_groups, num_ords*num_dofs);
  dealii::BlockVector<double> transported(num_groups, num_ords*num_dofs);
  for (int g = 0; g < num_groups; ++g) {
    d2m.vmult(flux_m.block(g), flux.block(g));
  }
  for (int g = 0; g < num_groups; ++g) {
    for (int down = 0; down < upscattering[g].size(); ++down) {
      upscattering[g][down].vmult(upscattered_m.block(g), 
                                  flux_m.block(g+1+down));
    }
  }
  for (int g = 0; g < num_groups; ++g) {
    m2d.vmult(upscattered.block(g), upscattered_m.block(g));
    within_groups[g].transport.vmult(transported.block(g), 
                                     upscattered.block(g));
  }
  // transported.sadd(-1, 1, src);
  transported += src;
  vmult(flux, transported);
}

template class FixedSourceGS<dealii::SolverGMRES<dealii::Vector<double>>, 1>;
template class FixedSourceGS<dealii::SolverGMRES<dealii::Vector<double>>, 2>;
template class FixedSourceGS<dealii::SolverGMRES<dealii::Vector<double>>, 3>;

}  // namespace aether::sn