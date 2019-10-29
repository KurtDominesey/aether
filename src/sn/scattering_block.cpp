#include "scattering_block.hpp"

template <int dim>
ScatteringBlock<dim>::ScatteringBlock(
    const Scattering<dim> &scattering, 
    const std::vector<double> &cross_sections)
    :  scattering(scattering), cross_sections(cross_sections) {}

template class ScatteringBlock<1>;
template class ScatteringBlock<2>;
template class ScatteringBlock<3>;