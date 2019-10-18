#include "within_group.hpp"

template <int dim, int qdim>
WithinGroup<dim, qdim>::WithinGroup(Transport<dim, qdim> &transport,
                                    MomentToDiscrete<qdim> &m2d,
                                    Scattering<dim> &scattering,
                                    DiscreteToMoment<qdim> &d2m)
    : transport(transport), m2d(m2d), scattering(scattering), d2m(d2m) {
  
}

template <int dim, int qdim>
void WithinGroup<dim, qdim>::vmult(
    dealii::BlockVector<double> &flux,
    const dealii::BlockVector<double> &src) const {
  AssertDimension(src.n_blocks(), flux.n_blocks());
  AssertDimension(src.size(), flux.size());
  // resize intermediate storage (could be cached in non-const method)
  const int num_dofs = src.block(0).size();
  dealii::BlockVector<double> src_m(1, num_dofs);
  dealii::BlockVector<double> scattered_m(1, num_dofs);
  dealii::BlockVector<double> scattered(src.n_blocks(), num_dofs);
  dealii::BlockVector<double> transported(src.n_blocks(), num_dofs);
  // apply the linear operator
  flux = 0;
  d2m.vmult(src_m, src);
  scattering.vmult(scattered_m, src_m);
  m2d.vmult(scattered, scattered_m);
  transport.vmult(transported, scattered);
  // transported /= 2;
  flux = 0;
  flux += src;
  flux -= transported;
  // std::cout << "UNCOLLIDED\n";
  // src.print(std::cout);
  // std::cout << "TRANSPORTED\n";
  // transported.print(std::cout);
  // std::cout << "FLUX\n";
  // flux.print(std::cout);
}