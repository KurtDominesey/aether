#ifndef AETHER_PGD_SN_WITHIN_MODE_H_
#define AETHER_PGD_SN_WITHIN_MODE_H_

#include <deal.II/lac/block_vector.h>

#include "pgd/sn/transport_block.h"
#include "sn/scattering_block.h"
#include "sn/moment_to_discrete.h"
#include "sn/discrete_to_moment.h"

namespace aether::pgd::sn {

/**
 * Within-mode operator for subspace PGD in space-angle.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class WithinMode {
 public:
  /**
   * Constructor.
   */
  WithinMode(const TransportBlock<dim, qdim> &transport,
             const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
             const aether::sn::ScatteringBlock<dim> &scattering,
             const aether::sn::DiscreteToMoment<dim, qdim> &d2m);

  /**
   * Constructor allowing unique pointers.
   */
  WithinMode(const std::unique_ptr<TransportBlock<dim, qdim>> &transport_unique,
             const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
             const std::unique_ptr<aether::sn::ScatteringBlock<dim>> &scattering_unique,
             const aether::sn::DiscreteToMoment<dim, qdim> &d2m);

  /**
   * Matrix-vector multplication by within-mode operator.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;

  /**
   * Adding matrix-vector multplication by within-mode operator.
   */
  void vmult_add(dealii::BlockVector<double> &dst,
                 const dealii::BlockVector<double> &src) const;

  //! Group transport block, \f$\underline{L}_g^{-1}(\overline{L}_g+B)\f$.
  const TransportBlock<dim, qdim> &transport;
  //! Within-group scattering block, \f$\Sigma_{s,g\rightarrow g}\f$.
  const aether::sn::ScatteringBlock<dim> &scattering;

 protected:
  //! Moment to discrete operator, \f$M\f$.
  const aether::sn::MomentToDiscrete<dim, qdim> &m2d;
  //! Discrete to moment operator, \f$D\f$.
  const aether::sn::DiscreteToMoment<dim, qdim> &d2m;
  //! Unique pointer to transport block.
  const std::unique_ptr<TransportBlock<dim, qdim>> transport_unique;
  //! Unique pointer to scattering block.
  const std::unique_ptr<aether::sn::ScatteringBlock<dim>> scattering_unique;

};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_WITHIN_MODE_H_