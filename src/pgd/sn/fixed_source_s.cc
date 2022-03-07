#include "fixed_source_s.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FixedSourceS<dim, qdim>::FixedSourceS(
    const std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> &blocks,
    const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
    const aether::sn::DiscreteToMoment<dim, qdim> &d2m,
    const Mgxs &mgxs)
    : blocks(blocks), m2d(m2d), d2m(d2m), mgxs(mgxs),
      streaming(blocks.size(), std::vector<double>(blocks.size())) {}

template <int dim, int qdim>
void FixedSourceS<dim, qdim>::vmult(dealii::Vector<double> &dst,
                                    const dealii::Vector<double> &src,
                                    bool transposing) const {
  const int num_modes = blocks.size();
  const int size = dst.size() / num_modes;
  AssertDimension(dst.size(), size * num_modes);
  AssertDimension(dst.size(), src.size());
  dealii::BlockVector<double> dst_b(num_modes, size);
  dealii::BlockVector<double> src_b(num_modes, size);
  dst_b = dst;
  src_b = src;
  vmult(dst_b, src_b, transposing);
  dst = dst_b;
}

template <int dim, int qdim>
void FixedSourceS<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src,
    bool transposing) const {
  transposing = transposing != transposed;  // (A^T)^T = A
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
  dealii::Vector<double> streamed(num_ordinates*num_dofs);
  dealii::Vector<double> collided(num_ordinates*num_dofs);
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
      streamed = 0;
      collided = 0;
      for (int mp = 0; mp < num_modes; ++mp) {
        int mpg = mp * num_groups + g;
        if (!transposing) {
          for (int down = 0; down < blocks[m][mp].upscattering[g].size(); 
               ++down)
            blocks[m][mp].upscattering[g][down].vmult_add(
                transferred_lm, src_lm.block(mpg+1+down));
          for (int up = 0; up < blocks[m][mp].downscattering[g].size(); ++up)
            blocks[m][mp].downscattering[g][up].vmult_add(
                transferred_lm, src_lm.block(mpg-1-up));
          blocks[m][mp].within_groups[g].scattering.vmult_add(
              transferred_lm, src_lm.block(mpg));
          streamed.add(streaming[m][mp], src.block(mpg));
          const auto &transport_block =
              dynamic_cast<const TransportBlock<dim, qdim>&>(
                blocks[m][mp].within_groups[g].transport);
          transport_block.collide_add(collided, src.block(mpg));
        } else {
          // TODO: test that this logic is correct for multigroup problems 
          for (int gp = 0; gp < g; ++gp) {
            int mpgp = mp * num_groups + gp;
            int gg = g - gp - 1;
            if (gg < blocks[mp][m].upscattering[gp].size())
              blocks[mp][m].upscattering[gp][gg].vmult_add(
                  transferred_lm, src_lm.block(mpgp));
          }
          for (int gp = g + 1; gp < num_groups; ++gp) {
            int mpgp = mp * num_groups + gp;
            int gg = gp - g - 1;
            if (gg < blocks[mp][m].downscattering[gp].size())
              blocks[mp][m].downscattering[gp][gg].vmult_add(
                  transferred_lm, src_lm.block(mpgp));
          }
          blocks[mp][m].within_groups[g].scattering.vmult_add(
              transferred_lm, src_lm.block(mpg));
          streamed.add(streaming[mp][m], src.block(mpg));
          const auto &transport_block = 
              dynamic_cast<const TransportBlock<dim, qdim>&>(
                blocks[mp][m].within_groups[g].transport);
          transport_block.collide_add(collided, src.block(mpg));
        }
      }
      const auto &transport_block = 
          dynamic_cast<const TransportBlock<dim, qdim>&>(
              blocks[m][m].within_groups[g].transport);
      transport_block.stream_add(dst.block(mg), streamed, transposing);
      transferred_lm *= -1;
      m2d.vmult_add(collided, transferred_lm);
      const auto &transport = 
          dynamic_cast<const Transport<dim, qdim>&>(transport_block.transport);
      transport.vmult_mass_add(dst.block(mg), collided);
    }
  }
}

template <int dim, int qdim>
void FixedSourceS<dim, qdim>::Tvmult(dealii::Vector<double> &dst,
                                     const dealii::Vector<double> &src) const {
  vmult(dst, src, true);
}

template <int dim, int qdim>
void FixedSourceS<dim, qdim>::Tvmult(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src) const {
  vmult(dst, src, true);
}

template <int dim, int qdim>
void FixedSourceS<dim, qdim>::get_inner_products_lhs(
    std::vector<std::vector<InnerProducts>> &inner_products,
    const dealii::BlockVector<double> &modes) {
  const int num_modes = inner_products.size();
  const int num_groups = modes.n_blocks() / num_modes;
  AssertDimension(modes.n_blocks(), num_modes*num_groups);
  const int num_ordinates = d2m.n_block_cols();
  const int num_dofs = modes.block(0).size() / num_ordinates;
  AssertDimension(modes.block(0).size(), num_ordinates*num_dofs);
  dealii::BlockVector<double> streamed(modes);
  dealii::BlockVector<double> modes_lm(modes.n_blocks(), num_dofs);
  dealii::BlockVector<double> scattered(modes);
  for (int m = 0, mg = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g, ++mg) {
      const auto &transport = dynamic_cast<const TransportBlock<dim, qdim>&>(
          blocks[m][m].within_groups[g].transport);
      transport.stream(streamed.block(mg), modes.block(mg));
      d2m.vmult(modes_lm.block(mg), modes.block(mg));
      m2d.vmult(scattered.block(mg), modes_lm.block(mg));
    }
    for (int mp = 0; mp < num_modes; ++mp)
      inner_products[m][mp] = 0;
  }
  const auto &transport = dynamic_cast<const Transport<dim, qdim>&>(
      blocks[0][0].within_groups[0].transport.transport);
  std::vector<dealii::types::global_dof_index> dof_indices(
      transport.dof_handler.get_fe().dofs_per_cell);
  unsigned int c = 0;
  using Cell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  for (Cell cell = transport.dof_handler.begin_active();
       cell != transport.dof_handler.end(); ++cell, ++c) {
    if (!cell->is_locally_owned())
      continue;
    const dealii::FullMatrix<double> &mass = transport.cell_matrices[c].mass;
    cell->get_dof_indices(dof_indices);
    int matl = cell->material_id();
    for (int m = 0, mg = 0; m < num_modes; ++m) {
      for (int g = 0; g < num_groups; ++g, ++mg) {
        for (int mp = 0; mp < num_modes; ++mp) {
          int mpg = mp * num_groups + g;
          for (int n = 0; n < transport.quadrature.size(); ++n) {
            double wn = transport.quadrature.weight(n);
            int nn = n * num_dofs;
            for (int i = 0; i < dof_indices.size(); ++i) {
              inner_products[m][mp].streaming += 
                  modes.block(mg)[nn+dof_indices[i]] *
                  streamed.block(mpg)[nn+dof_indices[i]] *
                  wn;
              for (int j = 0; j < dof_indices.size(); ++j) {
                inner_products[m][mp].collision[matl] += 
                    mgxs.total[g][matl] *
                    modes.block(mg)[nn+dof_indices[i]] *
                    mass[i][j] *
                    modes.block(mpg)[nn+dof_indices[j]] *
                    wn;
                for (int gp = 0; gp < num_groups; ++gp) {
                  int mpgp = mp * num_groups + gp;
                  inner_products[m][mp].scattering[matl][0] -=
                      mgxs.scatter[g][gp][matl] *
                      modes.block(mg)[nn+dof_indices[i]] *
                      mass[i][j] *
                      scattered.block(mpgp)[nn+dof_indices[j]] *
                      wn;
                  inner_products[m][mp].fission[matl] +=
                      mgxs.chi[g][matl] *
                      modes.block(mg)[nn+dof_indices[i]] *
                      mass[i][j] *
                      mgxs.nu_fission[gp][matl] *
                      scattered.block(mpgp)[nn+dof_indices[j]] *
                      wn;
                }
              }
            }
          }
        }
      }
    }
  }
}

template class FixedSourceS<1>;
template class FixedSourceS<2>;
template class FixedSourceS<3>;

}