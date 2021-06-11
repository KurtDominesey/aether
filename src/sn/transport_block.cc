#include "transport_block.h"

namespace aether::sn {

template <int dim, int qdim>
TransportBlock<dim, qdim>::TransportBlock(
    const Transport<dim, qdim> &transport, 
    const std::vector<double> &cross_sections,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions)
    : transport(transport),
      cross_sections(cross_sections),
      boundary_conditions(boundary_conditions) {
  boundary_conditions_zero.resize(boundary_conditions.size());
  for (int b = 0; b < boundary_conditions.size(); ++b)
    boundary_conditions_zero[b].reinit(boundary_conditions[b]);
}

template <int dim, int qdim>
int TransportBlock<dim, qdim>::n_block_cols() const {
  return transport.n_block_cols();
}

template <int dim, int qdim>
int TransportBlock<dim, qdim>::n_block_rows() const {
  return transport.n_block_rows();
}

template <int dim, int qdim>
int TransportBlock<dim, qdim>::m() const {
  return transport.m();
}

template <int dim, int qdim>
int TransportBlock<dim, qdim>::n() const {
  return transport.n();
}

template class TransportBlock<1>;
template class TransportBlock<2>;
template class TransportBlock<3>;

}  // namespace aether::sn