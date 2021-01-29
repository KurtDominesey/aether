#include "fixed_source_s_gs.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FixedSourceSGS<dim, qdim>::FixedSourceSGS(
    const Transport<dim, qdim> &transport,
    const aether::sn::Scattering<dim> &scattering,
    const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
    const aether::sn::DiscreteToMoment<dim, qdim> &d2m,
    const std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> &blocks,
    const std::vector<std::vector<double>> &streaming,
    const Mgxs &mgxs,
    const std::vector<std::vector<dealii::BlockVector<double>>>
      &boundary_conditions) 
    : m2d(m2d), d2m(d2m), blocks(blocks), streaming(streaming), 
      within_groups(blocks.size()), mgxs_wg(blocks.size(), mgxs) {
  const int num_modes = mgxs_wg.size();
  const int num_groups = mgxs_wg[0].total.size();
  for (int m = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g) {
      auto transport_wg = std::make_shared<TransportBlock<dim, qdim>>(
          transport, mgxs_wg[m].total[g], boundary_conditions[g]);
      auto scattering_wg = std::make_shared<aether::sn::ScatteringBlock<dim>>(
          scattering, mgxs_wg[m].scatter[g][g]);
      within_groups[m].emplace_back(transport_wg, m2d, scattering_wg, d2m);
    }
  }
}

template <int dim, int qdim>
void FixedSourceSGS<dim, qdim>::set_cross_sections(
    const std::vector<std::vector<Mgxs>> &mgxs) {
  const int num_modes = mgxs.size();
  const int num_groups = mgxs[0][0].total.size();
  const int num_materials = mgxs[0][0].total[0].size();
  for (int m = 0; m < num_modes; ++m) {
    mgxs_wg[m] = mgxs[m][m];
    for (int g = 0; g < num_groups; ++g) {
      for (int j = 0; j < num_materials; ++j) {
        AssertThrow(mgxs_wg[m].total[g][j] == mgxs[m][m].total[g][j],
                    dealii::ExcInvalidState());
        AssertThrow(mgxs_wg[m].scatter[g][g][j] == mgxs[m][m].scatter[g][g][j],
                    dealii::ExcInvalidState());
        mgxs_wg[m].total[g][j] /= streaming[m][m];
        mgxs_wg[m].scatter[g][g][j] /= streaming[m][m];
      }
    }
  }
}

template <int dim, int qdim>
void FixedSourceSGS<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst, 
    const dealii::BlockVector<double> &src) const {
  const int num_modes = blocks.size();
  const int num_groups = dst.n_blocks() / num_modes;
  AssertDimension(dst.n_blocks(), src.n_blocks());
  AssertDimension(dst.n_blocks(), num_modes*num_groups);
  const int num_ordinates = d2m.n_block_cols();
  const int num_dofs = dst.block(0).size() / num_ordinates;
  AssertDimension(dst.block(0).size(), num_ordinates*num_dofs);
  AssertDimension(dst.block(0).size(), src.block(0).size());
  const int num_qdofs = num_ordinates * num_dofs;
  dealii::BlockVector<double> dst_lm(num_modes*num_groups, num_dofs);
  dealii::Vector<double> uncollided(num_qdofs);
  dealii::Vector<double> operated(num_qdofs);
  dealii::Vector<double> streamed(num_qdofs);
  dealii::Vector<double> mass_inv(num_qdofs);
  dealii::Vector<double> scattered(num_dofs);
  dealii::IterationNumberControl control(10, 0);
  dealii::SolverRichardson<dealii::Vector<double>> solver(control);
  for (int m = 0, mg = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g, ++mg) {
      operated = 0;
      scattered = 0;
      streamed = 0;
      for (int mp = 0; mp <= m; ++mp) {
        int mpg = mp * num_groups + g;
        for (int up = 0; up < blocks[m][mp].downscattering[g].size(); ++up)
          blocks[m][mp].downscattering[g][up].vmult_add(
              scattered, dst_lm.block(mpg-1-up));
        if (m == mp)
          continue;
        for (int down = 0; down < blocks[m][mp].upscattering[g].size(); ++down)
          blocks[m][mp].upscattering[g][down].vmult_add(
              scattered, dst_lm.block(mpg+1+down));
        blocks[m][mp].within_groups[g].scattering.vmult_add(
            scattered, dst_lm.block(mpg));
        streamed.add(streaming[m][mp], dst.block(mpg));
        const auto &transport_block = 
            dynamic_cast<const TransportBlock<dim, qdim>&>(
              blocks[m][mp].within_groups[g].transport);
        transport_block.collide_add(operated, dst.block(mpg));
      }
      scattered *= -1;
      m2d.vmult_add(operated, scattered);
      const auto &transport_block = 
          dynamic_cast<const TransportBlock<dim, qdim>&>(
              blocks[m][m].within_groups[g].transport);
      mass_inv = src.block(mg);
      streamed *= -1;
      transport_block.stream_add(mass_inv, streamed);
      const auto &transport = 
          dynamic_cast<const Transport<dim, qdim>&>(transport_block.transport);
      transport.vmult_mass_inv(mass_inv);
      operated *= -1;
      operated += mass_inv;
      uncollided = 0;
      within_groups[m][g].transport.vmult(uncollided, operated);
      uncollided /= streaming[m][m];
      solver.solve(within_groups[m][g], dst.block(mg), uncollided,
                   dealii::PreconditionIdentity());
      d2m.vmult(dst_lm.block(mg), dst.block(mg));
    }
  }
}

template class FixedSourceSGS<1>;
template class FixedSourceSGS<2>;
template class FixedSourceSGS<3>;

}  // aether::pgd::sn