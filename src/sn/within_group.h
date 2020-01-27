#ifndef AETHER_SN_WITHIN_GROUP_H_
#define AETHER_SN_WITHIN_GROUP_H_

#include "transport_block.h"
#include "scattering_block.h"
#include "moment_to_discrete.h"
#include "discrete_to_moment.h"

namespace aether::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class WithinGroup {
 public:
  WithinGroup(TransportBlock<dim, qdim> &transport,
              MomentToDiscrete<qdim> &m2d,
              ScatteringBlock<dim> &scattering,
              DiscreteToMoment<qdim> &d2m);
  WithinGroup(std::shared_ptr<TransportBlock<dim, qdim>> &transport_shared,
              MomentToDiscrete<qdim> &m2d,
              std::shared_ptr<ScatteringBlock<dim>> &scattering_shared,
              DiscreteToMoment<qdim> &d2m);
  void vmult(dealii::Vector<double> &dst,
             const dealii::Vector<double> &src) const;
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  const TransportBlock<dim, qdim> &transport;
  const ScatteringBlock<dim> &scattering;

 protected:
  const MomentToDiscrete<qdim> &m2d;
  const DiscreteToMoment<qdim> &d2m;
  const std::shared_ptr<TransportBlock<dim, qdim>> transport_shared;
  const std::shared_ptr<ScatteringBlock<dim>> scattering_shared;
};

}  // namespace aether::sn

#endif  // AETHER_SN_WITHIN_GROUP_H_