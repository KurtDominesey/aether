#include "within_group.h"

namespace aether::sn {

template <int dim, int qdim>
WithinGroup<dim, qdim>::WithinGroup(TransportBlock<dim, qdim> &transport,
                                    MomentToDiscrete<qdim> &m2d,
                                    ScatteringBlock<dim> &scattering,
                                    DiscreteToMoment<qdim> &d2m)
    : transport(transport), m2d(m2d), 
      scattering(scattering), d2m(d2m) {}

template <int dim, int qdim>
WithinGroup<dim, qdim>::WithinGroup(
    std::shared_ptr<TransportBlock<dim, qdim>> &transport_shared,
    MomentToDiscrete<qdim> &m2d, 
    std::shared_ptr<ScatteringBlock<dim>> &scattering_shared,
    DiscreteToMoment<qdim> &d2m)
    : transport(*transport_shared.get()), 
      m2d(m2d), 
      scattering(*scattering_shared.get()), 
      d2m(d2m),
      transport_shared(transport_shared),
      scattering_shared(scattering_shared) {}

template <int dim, int qdim>
void WithinGroup<dim, qdim>::vmult(dealii::Vector<double> &flux,
                                   const dealii::Vector<double> &src) const {
  const int num_ords = transport.n_block_cols();
  const int num_dofs = flux.size() / num_ords;
  dealii::BlockVector<double> flux_b(num_ords, num_dofs);
  dealii::BlockVector<double> src_b(num_ords, num_dofs);
  src_b = src;
  vmult(flux_b, src_b);
  flux = flux_b;
}

template <int dim, int qdim>
void WithinGroup<dim, qdim>::vmult(
    dealii::BlockVector<double> &flux,
    const dealii::BlockVector<double> &src) const {
  AssertDimension(src.n_blocks(), flux.n_blocks());
  AssertDimension(src.size(), flux.size());
  // resize intermediate storage (could be cached in non-const method)
  // TODO: use GrowingVectorMemory
  const int num_dofs = src.block(0).size();
  dealii::BlockVector<double> src_m(1, num_dofs);
  dealii::BlockVector<double> scattered_m(1, num_dofs);
  dealii::BlockVector<double> scattered(src.n_blocks(), num_dofs);
  dealii::BlockVector<double> transported(src.n_blocks(), num_dofs);
  // apply the linear operator
  d2m.vmult(src_m, src);
  scattering.vmult(scattered_m, src_m);
  m2d.vmult(scattered, scattered_m);
  transported = src;
  transport.vmult(transported, scattered);  // L^-1 S x
  flux = src;  // I x
  flux -= transported;  // (I - L^-1 S) x
}

template class WithinGroup<1>;
template class WithinGroup<2>;
template class WithinGroup<3>;

}  // namespace aether::sn