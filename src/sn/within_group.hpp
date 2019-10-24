#ifndef AETHER_SN_WITHIN_GROUP_H_
#define AETHER_SN_WITHIN_GROUP_H_

#include <deal.II/lac/block_linear_operator.h>

#include "transport.hpp"
#include "scattering.hpp"
#include "moment_to_discrete.hpp"
#include "discrete_to_moment.hpp"

template <int dim, int qdim = dim == 1 ? 1 : 2>
class WithinGroup {
 public:
  WithinGroup(Transport<dim, qdim> &transport,
              MomentToDiscrete<qdim> &m2d,
              Scattering<dim> &scattering,
              DiscreteToMoment<qdim> &d2m);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;
 protected:
  const Transport<dim, qdim> &transport;
  const MomentToDiscrete<qdim> &m2d;
  const Scattering<dim> &scattering;
  const DiscreteToMoment<qdim> &d2m;
};

#endif  // AETHER_SN_WITHIN_GROUP_H_