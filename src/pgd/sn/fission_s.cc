#include "pgd/sn/fission_s.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FissionS<dim, qdim>::FissionS(
    const Transport<dim, qdim> &transport,
    const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
    const std::vector<std::vector<aether::sn::Emission<dim>>> &emission,
    const std::vector<std::vector<aether::sn::Production<dim>>> &production,
    const aether::sn::DiscreteToMoment<dim, qdim> &d2m)
    : transport(transport), m2d(m2d), emission(emission), 
      production(production), d2m(d2m) {}

template <int dim, int qdim>
void FissionS<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                const dealii::BlockVector<double> &src) const {
  const int num_modes = emission.size();
  AssertDimension(num_modes, production.size());
  const int num_groups = dst.n_blocks() / num_modes;
  AssertDimension(dst.n_blocks(), num_modes*num_groups);
  AssertDimension(dst.n_blocks(), src.n_blocks());
  const int num_ordinates = d2m.n_block_cols();
  const int num_dofs = dst.block(0).size() / num_ordinates;
  AssertDimension(dst.block(0).size(), num_ordinates*num_dofs);
  dealii::BlockVector<double> src_lm(num_modes*num_groups, num_dofs);
  dealii::BlockVector<double> src_mp(num_groups, num_dofs);
  dealii::BlockVector<double> emitted(num_groups, num_dofs);
  dealii::Vector<double> produced(num_dofs);
  dealii::Vector<double> produced_mass(num_dofs);
  for (int m = 0, mg = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g, ++mg) {
      d2m.vmult(src_lm.block(mg), src.block(mg));
    }
  }
  dst = 0;
  for (int m = 0; m < num_modes; ++m) {
    emitted = 0;
    for (int mp = 0; mp < num_modes; ++mp) {
      for (int g = 0; g < num_groups; ++g) {
        src_mp.block(g) = src_lm.block(mp*num_groups+g);
      }
      production[m][mp].vmult(produced, src_mp);
      transport.collide_ordinate(produced_mass, produced);
      emission[m][mp].vmult_add(emitted, produced_mass);
    }
    for (int g = 0; g < num_groups; ++g) {
      m2d.vmult_add(dst.block(m*num_groups+g), emitted.block(g));
    }
  }
}

template class FissionS<1>;
template class FissionS<2>;
template class FissionS<3>;

}