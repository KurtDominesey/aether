#include "scattering_block.hpp"

template <int dim>
ScatteringBlock<dim>::ScatteringBlock(
    const Scattering<dim> &scattering, 
    const std::vector<double> &cross_sections)
    :  scattering(scattering), cross_sections(cross_sections) {}

template <int dim>
template <typename VectorType>
void ScatteringBlock<dim>::vmult(VectorType &dst, const VectorType &src) 
    const {
  scattering.vmult(dst, src, cross_sections);
}

template <int dim>
template <typename VectorType>
void ScatteringBlock<dim>::vmult_add(VectorType &dst, const VectorType &src) 
    const {
  scattering.vmult_add(dst, src, cross_sections);
}