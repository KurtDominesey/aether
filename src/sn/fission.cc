#include "fission.h"

namespace aether::sn {

template <int dim, int qdim>
Fission<dim, qdim>::Fission(
    const std::vector<WithinGroup<dim, qdim>> &within_groups,
    const MomentToDiscrete<dim, qdim> &m2d,
    const Emission<dim> &emission,
    const Production<dim> &production,
    const DiscreteToMoment<dim, qdim> &d2m) 
    : m2d(m2d), emission(emission), production(production), d2m(d2m) {
  for (int g = 0; g < within_groups.size(); ++g)
    transport_blocks.push_back(within_groups[g].transport);
}

template <int dim, int qdim>
void Fission<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                               const dealii::BlockVector<double> &src,
                               const bool sweep,
                               bool transposing) const {
  transposing = transposing != transposed;  // (A^T)^T = A
  AssertDimension(dst.n_blocks(), src.n_blocks());
  AssertDimension(dst.size(), src.size());
  const int num_ords = d2m.n_block_cols();
  const int num_dofs = src.block(0).size() / num_ords;
  const int num_groups = transport_blocks.size();
  dealii::Vector<double> produced(num_dofs);
  dealii::BlockVector<double> scratch(num_groups, num_dofs);
  for (int g = 0; g < num_groups; ++g)
    d2m.vmult(scratch.block(g), src.block(g));
  if (!transposing) {
    production.vmult(produced, scratch);
    emission.vmult(scratch, produced);
  } else {
    emission.Tvmult(produced, scratch);
    production.Tvmult(scratch, produced);
  }
  dealii::Vector<double> emitted(num_ords*num_dofs);
  dst = 0;
  for (int g = 0; g < num_groups; ++g) {
    m2d.vmult(emitted, scratch.block(g));
    if (sweep) {
      if (!transposing)
        transport_blocks[g].get().vmult(dst.block(g), emitted, true);
      else
        transport_blocks[g].get().Tvmult(dst.block(g), emitted, true);
    }
    else
      dst.block(g) = emitted;
  }
}

template class Fission<1>;
template class Fission<2>;
template class Fission<3>;

}  // namespace aether::sn