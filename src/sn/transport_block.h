#ifndef AETHER_SN_TRANSPORT_BLOCK_H_
#define AETHER_SN_TRANSPORT_BLOCK_H_

#include "transport.h"

namespace aether::pgd::sn {
template <int dim, int qdim> class FixedSourceP;
}

namespace aether::sn {

/**
 * A group-specific transport operator \f$L^{-1}_g\f$ with boundary conditions
 * given by \f$\psi_\text{inc}\f$.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
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
      const Transport<dim, qdim> &transport,
      const std::vector<double> &cross_sections,
      const std::vector<dealii::BlockVector<double>> &boundary_conditions);

  virtual ~TransportBlock() {}

  /**
   * Apply the linear operator.
   * 
   * @param dst The destination vector.
   * @param src The source vector.
   * @param homogeneous Whether to ignore the inhomogeneous boundary conditions.
   */
  template <typename VectorType>
  void vmult(VectorType &dst, const VectorType &src,
             const bool homogeneous = true) const;

  /**
   * Return the number of blocks in a column.
   */
  int n_block_cols() const;

  /**
   * Return the number of blocks in a row.
   */
  int n_block_rows() const;

  //! The transport operator.
  const Transport<dim> &transport;

 protected:
  //! The total material cross-sections.
  const std::vector<double> &cross_sections;
  //! The values of \f$\psi_\text{inc}\f$ by boundary id.
  const std::vector<dealii::BlockVector<double>> &boundary_conditions;
  //! Zero values of \f$\psi_\text{inc}\f$ by boundary id. 
  std::vector<dealii::BlockVector<double>> boundary_conditions_zero;

  friend class aether::pgd::sn::FixedSourceP<dim, qdim>;
};

template <int dim, int qdim>
template <typename VectorType>
void TransportBlock<dim, qdim>::vmult(VectorType &dst, const VectorType &src,
                                      const bool homogeneous) const {
  if (homogeneous)
    transport.vmult(dst, src, cross_sections, boundary_conditions_zero);
  else
    transport.vmult(dst, src, cross_sections, boundary_conditions);
}

}

#endif // AETHER_SN_TRANSPORT_BLOCK_H_