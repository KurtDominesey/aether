#ifndef AETHER_SN_SCATTERING_BLOCK_H_
#define AETHER_SN_SCATTERING_BLOCK_H_

#include "scattering.hpp"

/**
 * A specific, group-to-group scattering operator, \f$S_{g'\rightarrow g}\f$.
 */
template <int dim>
class ScatteringBlock {
 public:
  /**
   * Constructor.
   * 
   * @param scattering The scattering operator.
   * @param cross_section The material scattering cross-sections.
   */
  ScatteringBlock(const Scattering<dim> &scattering,
                  const std::vector<double> &cross_sections);

  /**
   * Apply the linear operator.
   * 
   * @param dst The destination vector.
   * @param src The source vector.
   */
  template <typename VectorType>
  void vmult(VectorType &dst, const VectorType &src) const;

  /**
   * Apply and add the linear operator.
   * 
   * @param dst The destination vector.
   * @param src The source vector.
   */
  template <typename VectorType>
  void vmult_add(VectorType &dst, const VectorType &src) const;

 protected:
  //! The scattering operator.
  const Scattering<dim> &scattering;
  //! The material scattering cross-sections.
  const std::vector<double> &cross_sections;
};

template <int dim>
template <typename VectorType>
void ScatteringBlock<dim>::vmult(VectorType &dst, const VectorType &src) 
    const {
  scattering.vmult(dst, src, cross_sections);
}

template <int dim>
template <typename VectorType>
void ScatteringBlock<dim>::vmult_add(VectorType &dst, 
                                            const VectorType &src) const {
  scattering.vmult_add(dst, src, cross_sections);
}

#endif  // AETHER_SN_SCATTERING_BLOCK_H_