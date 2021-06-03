#include "fixed_source.h"

namespace aether::sn {

template <int dim, int qdim>
FixedSource<dim, qdim>::FixedSource(
    std::vector<WithinGroup<dim, qdim>> &within_groups,
    std::vector<std::vector<ScatteringBlock<dim>>> &downscattering,
    std::vector<std::vector<ScatteringBlock<dim>>> &upscattering,
    MomentToDiscrete<dim, qdim> &m2d, 
    DiscreteToMoment<dim, qdim> &d2m)
    : within_groups(within_groups),
      downscattering(downscattering),
      upscattering(upscattering),
      m2d(m2d),
      d2m(d2m) {}

template <int dim, int qdim>
void FixedSource<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src,
    bool transposing) const {
  transposing = transposing != transposed;  // (A^T)^T = A
  const int num_groups = within_groups.size();
  const int num_ords = within_groups[0].transport.n_block_cols();
  const int num_dofs = dst.block(0).size() / num_ords;
  AssertDimension(num_groups, dst.n_blocks());
  AssertDimension(num_groups, src.n_blocks());
  AssertDimension(num_groups, upscattering.size());
  AssertDimension(num_groups, downscattering.size());
  dst = src;  // I x
  dealii::BlockVector<double> src_m(num_groups, num_dofs);
  for (int g = 0; g < num_groups; ++g)
    d2m.vmult(src_m.block(g), src.block(g));  // D x
  dealii::Vector<double> inscattered_m(num_dofs);
  dealii::Vector<double> inscattered(num_ords*num_dofs);
  dealii::Vector<double> transported(num_ords*num_dofs);
  for (int g = 0; g < num_groups; ++g) {
    Assert(upscattering[g].size() < num_groups - g, dealii::ExcInvalidState());
    Assert(downscattering[g].size() < g + 1, dealii::ExcInvalidState());
    transported = 0;
    inscattered_m = 0;
    if (!transposing) {
      for (int down = 0; down < upscattering[g].size(); ++down)
        upscattering[g][down].vmult_add(inscattered_m, src_m.block(g+1+down));
      for (int up = 0; up < downscattering[g].size(); ++up)
        downscattering[g][up].vmult_add(inscattered_m, src_m.block(g-1-up));
    } else {
      for (int gp = 0; gp < g; ++gp) {
        int gg = g - gp - 1; 
        if (gg < upscattering[gp].size())
          upscattering[gp][gg].vmult_add(inscattered_m, src_m.block(gp));
      }
      for (int gp = g + 1; gp < num_groups; ++gp) {
        int gg = gp - g - 1;
        if (gg < downscattering[gp].size())
          downscattering[gp][gg].vmult_add(inscattered_m, src_m.block(gp));
      }
    }
    within_groups[g].scattering.vmult_add(inscattered_m, src_m.block(g));
    m2d.vmult(inscattered, inscattered_m);  // M S D x
    transported = src.block(g);
    if (!transposing)  // L^-1 M S D x
      within_groups[g].transport.vmult(transported, inscattered);
    else  // (L^T)^-1 M S D x
      within_groups[g].transport.Tvmult(transported, inscattered);
    dst.block(g) -= transported;  // (I - L^-1 M S D) x
  }
}

template <int dim, int qdim>
int FixedSource<dim, qdim>::m() const {
  return within_groups.size() * within_groups[0].transport.transport.m();
}

template <int dim, int qdim>
int FixedSource<dim, qdim>::n() const {
  return m();
}

template class FixedSource<1>;
template class FixedSource<2>;
template class FixedSource<3>;

}  // namespace aether::sn