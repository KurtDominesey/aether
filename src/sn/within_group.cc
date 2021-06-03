#include "within_group.h"

namespace aether::sn {

template <int dim, int qdim>
WithinGroup<dim, qdim>::WithinGroup(const TransportBlock<dim, qdim> &transport,
                                    const MomentToDiscrete<dim, qdim> &m2d,
                                    const ScatteringBlock<dim> &scattering,
                                    const DiscreteToMoment<dim, qdim> &d2m)
    : transport(transport), m2d(m2d), 
      scattering(scattering), d2m(d2m) {}

template <int dim, int qdim>
WithinGroup<dim, qdim>::WithinGroup(
    const std::shared_ptr<TransportBlock<dim, qdim>> &transport_shared,
    const MomentToDiscrete<dim, qdim> &m2d, 
    const std::shared_ptr<ScatteringBlock<dim>> &scattering_shared,
    const DiscreteToMoment<dim, qdim> &d2m)
    : transport(*transport_shared.get()), 
      m2d(m2d), 
      scattering(*scattering_shared.get()), 
      d2m(d2m),
      transport_shared(transport_shared),
      scattering_shared(scattering_shared) {}

template <int dim, int qdim>
void WithinGroup<dim, qdim>::vmult(dealii::Vector<double> &flux,
                                   const dealii::Vector<double> &src,
                                   const bool transposing) const {
  const int num_ords = transport.n_block_cols();
  const int num_dofs = flux.size() / num_ords;
  dealii::BlockVector<double> flux_b(num_ords, num_dofs);
  dealii::BlockVector<double> src_b(num_ords, num_dofs);
  src_b = src;
  vmult(flux_b, src_b, transposing);
  flux = flux_b;
}

template <int dim, int qdim>
void WithinGroup<dim, qdim>::vmult(
    dealii::BlockVector<double> &flux,
    const dealii::BlockVector<double> &src,
    const bool transposing) const {
  const int num_dofs = src.block(0).size();
  dealii::BlockVector<double> src_m(1, num_dofs);
  dealii::BlockVector<double> scattered_m(1, num_dofs);
  dealii::BlockVector<double> scattered(src.n_blocks(), num_dofs);
  dealii::BlockVector<double> transported(src.n_blocks(), num_dofs);
  vmult(flux, src, src_m, scattered_m, scattered, transported, transposing);
}

template <int dim, int qdim>
void WithinGroup<dim, qdim>::vmult(
      dealii::PETScWrappers::MPI::BlockVector &flux,
      const dealii::PETScWrappers::MPI::BlockVector &src,
      const bool transposing) const {
  const MPI_Comm& communicator = src.get_mpi_communicator();
  const int size = src.block(0).size();
  const int local_size = src.block(0).local_size();
  using BlockVector = dealii::PETScWrappers::MPI::BlockVector;
  BlockVector src_m(1, communicator, size, local_size);
  BlockVector scattered_m(1, communicator, size, local_size);
  BlockVector scattered(src.n_blocks(), communicator, size, local_size);
  BlockVector transported(src.n_blocks(), communicator, size, local_size);
  vmult(flux, src, src_m, scattered_m, scattered, transported, transposing);
}

template <int dim, int qdim>
template <class Vector>
void WithinGroup<dim, qdim>::vmult(
    dealii::BlockVectorBase<Vector> &flux,
    const dealii::BlockVectorBase<Vector> &src,
    dealii::BlockVectorBase<Vector> &src_m,
    dealii::BlockVectorBase<Vector> &scattered_m,
    dealii::BlockVectorBase<Vector> &scattered,
    dealii::BlockVectorBase<Vector> &transported,
    bool transposing) const {
  transposing = transposing != transposed;  // (A^T)^T = A
  AssertDimension(src.n_blocks(), flux.n_blocks());
  AssertDimension(src.size(), flux.size());
  // apply the linear operator
  d2m.vmult(src_m, src);
  scattering.vmult(scattered_m, src_m);
  m2d.vmult(scattered, scattered_m);
  transported = src;
  if (!transposing)
    transport.vmult(transported, scattered);  // L^-1 S x
  else
    transport.Tvmult(transported, scattered);  // (L^T)^-1 S x
  flux = src;  // I x
  flux -= transported;  // (I - L^-1 S) x
}

template class WithinGroup<1>;
template class WithinGroup<2>;
template class WithinGroup<3>;

}  // namespace aether::sn