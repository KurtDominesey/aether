#ifndef AETHER_SN_WITHIN_GROUP_H_
#define AETHER_SN_WITHIN_GROUP_H_

#include "transport_block.h"
#include "scattering_block.h"
#include "moment_to_discrete.h"
#include "discrete_to_moment.h"

namespace aether::sn {

/**
 * Within-group operator.
 * 
 * Implements \f$I+\underline{L}_g^{-1}(\overline{L}_g-BM
 * \Sigma_{s,g\rightarrow g}D)\f$.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class WithinGroup {
 public:
  /**
   * Constructor.
   */
  WithinGroup(const TransportBlock<dim, qdim> &transport,
              const MomentToDiscrete<dim, qdim> &m2d,
              const ScatteringBlock<dim> &scattering,
              const DiscreteToMoment<dim, qdim> &d2m);
  /**
   * Constructor allowing shared pointers.
   */
  WithinGroup(const std::shared_ptr<TransportBlock<dim, qdim>> &transport_shared,
              const MomentToDiscrete<dim, qdim> &m2d,
              const std::shared_ptr<ScatteringBlock<dim>> &scattering_shared,
              const DiscreteToMoment<dim, qdim> &d2m);
  /**
   * Matrix-vector multiplication by within-group operator.
   */
  void vmult(dealii::Vector<double> &dst,
             const dealii::Vector<double> &src) const;
  /**
   * Matrix-vector multplication by within-group operator.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  //! Group transport block, \f$\underline{L}_g^{-1}(\overline{L}_g+B)\f$.
  const TransportBlock<dim, qdim> &transport;
  //! Within-group scattering block, \f$\Sigma_{s,g\rightarrow g}\f$.
  const ScatteringBlock<dim> &scattering;

 protected:
  //! Moment to discrete operator, \f$M\f$.
  const MomentToDiscrete<dim, qdim> &m2d;
  //! Discrete to moment operator, \f$D\f$.
  const DiscreteToMoment<dim, qdim> &d2m;
  //! Shared pointer to transport block.
  const std::shared_ptr<TransportBlock<dim, qdim>> transport_shared;
  //! Shared pointer to scattering block.
  const std::shared_ptr<ScatteringBlock<dim>> scattering_shared;
};

}  // namespace aether::sn

#endif  // AETHER_SN_WITHIN_GROUP_H_