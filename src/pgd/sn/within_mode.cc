#include "pgd/sn/within_mode.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
WithinMode<dim, qdim>::WithinMode(
    const TransportBlock<dim, qdim> &transport,
    const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
    const aether::sn::ScatteringBlock<dim> &scattering,
    const aether::sn::DiscreteToMoment<dim, qdim> &d2m)
    : transport(transport), m2d(m2d), scattering(scattering), d2m(d2m) {}

template <int dim, int qdim>
WithinMode<dim, qdim>::WithinMode(
    const std::unique_ptr<TransportBlock<dim, qdim>> &transport_unique,
    const aether::sn::MomentToDiscrete<dim, qdim> &m2d, 
    const std::unique_ptr<aether::sn::ScatteringBlock<dim>> &scattering_unique,
    const aether::sn::DiscreteToMoment<dim, qdim> &d2m)
    : transport(*transport_unique), 
      m2d(m2d), 
      scattering(*scattering_unique), 
      d2m(d2m),
      transport_unique(transport_unique),
      scattering_unique(scattering_unique) {}

template <int dim, int qdim>
void WithinMode<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                  const dealii::BlockVector<double> &src)
                                  const {
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int qdim>
void WithinMode<dim, qdim>::vmult_add(dealii::BlockVector<double> &dst, 
                                      const dealii::BlockVector<double> &src)
                                      const {
  const int num_dofs = src.block(0).size();
  dealii::BlockVector<double> src_m(1, num_dofs);
  dealii::BlockVector<double> scattered_m(1, num_dofs);
  transport.stream_add(dst, src);
  transport.collide_add(dst, src, true);
  d2m.vmult(src_m, src);
  scattering.vmult(scattered_m, src_m);
  scattered_m *= -1;
  m2d.vmult_add(dst, scattered_m);
}

}