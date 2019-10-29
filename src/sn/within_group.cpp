#include "within_group.hpp"

template <int dim, int qdim>
WithinGroup<dim, qdim>::WithinGroup(Transport<dim, qdim> &transport,
                                    MomentToDiscrete<qdim> &m2d,
                                    Scattering<dim> &scattering,
                                    DiscreteToMoment<qdim> &d2m)
    : transport(std::move(transport)), m2d(m2d), 
      scattering(std::move(scattering)), d2m(d2m) {}


template <int dim, int qdim>
void WithinGroup<dim, qdim>::vmult(dealii::Vector<double> &flux,
                                   const dealii::Vector<double> &src) const {
  const int num_ords = transport.quadrature.size();
  const int num_dofs = transport.dof_handler.n_dofs();
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
  transport.vmult(transported, scattered);  // L^-1 S x
  flux = src;  // I x
  flux -= transported;  // (I - L^-1 S) x
}

template class WithinGroup<1>;
template class WithinGroup<2>;
template class WithinGroup<3>;