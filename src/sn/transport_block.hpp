#ifndef AETHER_SN_TRANSPORT_BLOCK_H_
#define AETHER_SN_TRANSPORT_BLOCK_H_

#include "transport.hpp"

/**
 * A group-specific transport operator \f$L^{-1}_g\f$ with boundary conditions
 * given by \f$\psi_\text{inc}\f$.
 */
template <int dim>
class TransportBlock {
 public:
  /**
   * Constructor.
   * 
   * @param transport The transport operator.
   * @param cross_sections The total material cross-sections.
   * @param boundary_conditions The values of \f$\psi_\text{inc}\f$ by 
   *                            boundary id.
   */
  TransportBlock(
      const Transport<dim> &transport,
      const std::vector<double> &cross_sections,
      const std::vector<dealii::BlockVector<double>> &boundary_conditions);

  /**
   * Apply the linear operator.
   * 
   * @param dst The destination vector.
   * @param src The source vector.
   * @param homogeneous Whether to ignore the inhomogeneous boundary conditions.
   */
  template <typename VectorType>
  void vmult(VectorType &dst, const VectorType &src,
             const bool homogeneous) const;

 protected:
  //! The transport operator.
  const Transport<dim> &transport;
  //! The total material cross-sections.
  const std::vector<double> &cross_sections;
  //! The values of \f$\psi_\text{inc}\f$ by boundary id.
  const std::vector<dealii::BlockVector<double>> &boundary_conditions;
  //! Zero values of \f$\psi_\text{inc}\f$ by boundary id. 
  std::vector<dealii::BlockVector<double>> boundary_conditions_zero;
};

template <int dim>
template <typename VectorType>
void TransportBlock<dim>::vmult(VectorType &dst, const VectorType &src,
                                const bool homogeneous) const {
  if (homogeneous)
    transport.vmult(dst, src, cross_sections, boundary_conditions_zero);
  else
    transport.vmult(dst, src, cross_sections, boundary_conditions);
}

#endif // AETHER_SN_TRANSPORT_BLOCK_H_