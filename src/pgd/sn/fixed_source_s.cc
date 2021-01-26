#include "fixed_source_s.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FixedSourceS<dim, qdim>::FixedSourceS(
    const std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> &blocks,
    const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
    const aether::sn::DiscreteToMoment<dim, qdim> &d2m)
    : blocks(blocks), m2d(m2d), d2m(d2m), 
      streaming(blocks.size(), std::vector<double>(blocks.size())) {}

template <int dim, int qdim>
void FixedSourceS<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src) const {
  AssertDimension(dst.size(), src.size());
  AssertDimension(dst.n_blocks(), src.n_blocks());
  const int num_groups = blocks[0][0].within_groups.size();
  const int num_modes = dst.n_blocks() / num_groups;
  AssertDimension(dst.n_blocks(), num_modes*num_groups);
  const int num_ordinates = d2m.n_block_cols();
  const int num_dofs = dst.block(0).size() / num_ordinates;
  AssertDimension(dst.block(0).size(), num_ordinates*num_dofs);
  dealii::BlockVector<double> src_lm(num_modes*num_groups, num_dofs);
  dealii::Vector<double> transferred_lm(num_dofs);
  dealii::Vector<double> transferred(num_ordinates*num_dofs);
  dealii::Vector<double> transferred_dual(num_ordinates*num_dofs);
  dealii::Vector<double> transported(num_ordinates*num_dofs);
  dealii::Vector<double> streamed(num_ordinates*num_dofs);
  dst = 0;
  dealii::Vector<double> src_dual_mg(src.block(0).size());
  for (int m = 0, mg = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g, ++mg) {
      d2m.vmult(src_lm.block(mg), src.block(mg));
    }
  }
  for (int m = 0, mg = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g, ++mg) {
      transferred_lm = 0;
      transferred = 0;
      for (int mp = 0; mp < num_modes; ++mp) {
        int mpg = mp * num_groups + g;
        for (int down = 0; down < blocks[m][mp].upscattering[g].size(); ++down)
          blocks[m][mp].upscattering[g][down].vmult_add(
              transferred_lm, src_lm.block(mpg+1+down));
        for (int up = 0; up < blocks[m][mp].downscattering[g].size(); ++up)
          blocks[m][mp].downscattering[g][up].vmult_add(
              transferred_lm, src_lm.block(mpg-1-up));
        blocks[m][mp].within_groups[g].scattering.vmult_add(
            transferred_lm, src_lm.block(mpg));
        
        const auto &transport = dynamic_cast<const TransportBlock<dim, qdim>&>(
            blocks[m][mp].within_groups[g].transport);
        streamed = 0;
        transport.stream(streamed, src.block(mpg));
        streamed *= streaming[m][mp];
        transport.collide_add(streamed, src.block(mpg));
        dst.block(mg) += streamed;
      }
      m2d.vmult(transferred, transferred_lm);
      const auto &transport = dynamic_cast<const Transport<dim, qdim>&>(
          blocks[m][m].within_groups[g].transport.transport);
      transport.vmult_mass(transferred_dual, transferred);
      dst.block(mg) -= transferred_dual;
    }
  }
}

template <int dim, int qdim>
void FixedSourceS<dim, qdim>::get_inner_products_lhs(
    std::vector<std::vector<InnerProducts>> &inner_products,
    const dealii::BlockVector<double> &modes) {
  const int num_modes = inner_products.size();
  const int num_groups = modes.n_blocks() / num_modes;
  AssertDimension(modes.n_blocks(), num_modes*num_groups);
  dealii::Vector<double> scratch(modes.block(0).size());
  dealii::Vector<double> scattered;
  for (int m = 0; m < num_modes; ++m) {
    for (int mp = 0; mp < num_modes; ++m) {
      for (int g = 0; g < num_groups; ++g) {
        
      }
    }
  }
}

template class FixedSourceS<1>;
template class FixedSourceS<2>;
template class FixedSourceS<3>;

}