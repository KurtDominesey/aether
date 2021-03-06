#include "fixed_source_gs.h"

namespace aether::sn {

template <class SolverType, int dim, int qdim>
FixedSourceGS<SolverType, dim, qdim>::FixedSourceGS(
      const std::vector<WithinGroup<dim, qdim>> &within_groups,
      const std::vector<std::vector<ScatteringBlock<dim>>> &downscattering,
      const std::vector<std::vector<ScatteringBlock<dim>>> &upscattering,
      const MomentToDiscrete<dim, qdim> &m2d,
      const DiscreteToMoment<dim, qdim> &d2m,
      SolverType &solver)
      : within_groups(within_groups),
        downscattering(downscattering),
        upscattering(upscattering),
        m2d(m2d), 
        d2m(d2m),
        solver(solver) {}

template <class SolverType, int dim, int qdim>
FixedSourceGS<SolverType, dim, qdim>::FixedSourceGS(
      const FixedSource<dim, qdim> &fixed_source, SolverType &solver)
      : within_groups(fixed_source.within_groups),
        downscattering(fixed_source.downscattering),
        upscattering(fixed_source.upscattering),
        m2d(fixed_source.m2d), 
        d2m(fixed_source.d2m),
        solver(solver) {
  transposed = fixed_source.transposed;
}

template <class SolverType, int dim, int qdim>
void FixedSourceGS<SolverType, dim, qdim>::vmult(
      dealii::BlockVector<double> &dst,
      const dealii::BlockVector<double> &src) const {
  if (transposed)
    do_Tvmult(dst, src);
  else
    do_vmult(dst, src);
}

template <class SolverType, int dim, int qdim>
void FixedSourceGS<SolverType, dim, qdim>::Tvmult(
      dealii::BlockVector<double> &dst,
      const dealii::BlockVector<double> &src) const {
  if (transposed)
    do_vmult(dst, src);  // (A^T)^T = A
  else
    do_Tvmult(dst, src);
}

template <class SolverType, int dim, int qdim>
void FixedSourceGS<SolverType, dim, qdim>::do_vmult(
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
    try {
      solver.solve(within_groups[g], dst.block(g), transported,
                  dealii::PreconditionIdentity());
    } catch (dealii::SolverControl::NoConvergence &failure) {
      // TODO: log this event, don't print
      std::cout << "Within-group failure in group " << g << std::endl;
      failure.print_info(std::cout);
    }
    d2m.vmult(dst_m.block(g), dst.block(g));
  }
}

template <class SolverType, int dim, int qdim>
void FixedSourceGS<SolverType, dim, qdim>::do_Tvmult(
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
  dealii::Vector<double> upscattered_m(num_dofs);
  dealii::Vector<double> upscattered(num_ords*num_dofs);
  dealii::Vector<double> transported(num_ords*num_dofs);
  for (int g = num_groups-1; g >= 0; --g) {
    Assert(downscattering[g].size() < g + 1, dealii::ExcInvalidState());
    transported = 0;
    upscattered_m = 0;
    for (int gp = g+1; gp < num_groups; ++gp) {
      int gg  = gp - g - 1;
      // forward downscattering becomes adjoint upscattering
      if (gg < downscattering[gp].size())
        downscattering[gp][gg].vmult_add(upscattered_m, dst_m.block(gp));
    }
    m2d.vmult(upscattered, upscattered_m);
    within_groups[g].transport.Tvmult(transported, upscattered);
    transported += src.block(g);
    try {
      const auto within_group = dealii::linear_operator(within_groups[g]);
      const auto within_groupT = dealii::transpose_operator(within_group);
      solver.solve(within_groupT, dst.block(g), transported,
                  dealii::PreconditionIdentity());
    } catch (dealii::SolverControl::NoConvergence &failure) {
      // TODO: log this event, don't print
      std::cout << "Within-group failure in group " << g << std::endl;
      failure.print_info(std::cout);
    }
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

template class FixedSourceGS<dealii::SolverRichardson<dealii::Vector<double>>, 1>;
template class FixedSourceGS<dealii::SolverRichardson<dealii::Vector<double>>, 2>;
template class FixedSourceGS<dealii::SolverRichardson<dealii::Vector<double>>, 3>;

}  // namespace aether::sn