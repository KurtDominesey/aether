#include "transport_block.hpp"

template <int dim>
TransportBlock<dim>::TransportBlock(
    const Transport<dim> &transport, 
    const std::vector<double> &cross_sections,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions)
    : transport(transport),
      cross_sections(cross_sections),
      boundary_conditions(boundary_conditions) {}

template <int dim>
template <typename VectorType>
void TransportBlock<dim>::vmult(VectorType &dst, const VectorType &src,
                                bool homogeneous) const {
  if (homogeneous)
    transport.vmult(dst, src, cross_sections);
  else
    transport.vmult(dst, src, cross_sections, boundary_conditions);
}