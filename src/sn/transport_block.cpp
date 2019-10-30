#include "transport_block.hpp"

template <int dim>
TransportBlock<dim>::TransportBlock(
    const Transport<dim> &transport, 
    const std::vector<double> &cross_sections,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions)
    : transport(transport),
      cross_sections(cross_sections),
      boundary_conditions(boundary_conditions) {
  boundary_conditions_zero.resize(boundary_conditions.size());
  for (int b = 0; b < boundary_conditions.size(); ++b)
    boundary_conditions_zero[b].reinit(boundary_conditions[b]);
}