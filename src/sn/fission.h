#ifndef AETHER_SN_FISSION_H_
#define AETHER_SN_FISSION_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include "sn/transport_block.h"
#include "sn/within_group.h"
#include "sn/moment_to_discrete.h"
#include "sn/emission.h"
#include "sn/production.h"
#include "sn/discrete_to_moment.h"

namespace aether::sn {

/**
 * Uncollided fission neutron source.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class Fission {
 public:
  using TransportBlocks =
      std::vector<std::reference_wrapper<const TransportBlock<dim, qdim>>>;
  Fission(const std::vector<WithinGroup<dim, qdim>> &within_groups,
          const MomentToDiscrete<dim, qdim> &m2d,
          const Emission<dim> &emission,
          const Production<dim> &production,
          const DiscreteToMoment<dim, qdim> &d2m);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src,
             const bool sweep=true, bool transposing=false) const;
  template <typename VectorType>
  void Tvmult(VectorType &dst, const VectorType &src, 
              const bool sweep=true) const;
  bool transposed = false;

 protected:
  TransportBlocks transport_blocks;
  const MomentToDiscrete<dim, qdim> &m2d;
  const Emission<dim> &emission;
  const Production<dim> &production;
  const DiscreteToMoment<dim, qdim> d2m;
};

template <int dim, int qdim>
template <typename VectorType>
void Fission<dim, qdim>::Tvmult(VectorType &dst, const VectorType &src, 
                                const bool sweep) const {
  vmult(dst, src, sweep, true);
}

}  // namespace aether::sn

#endif  // AETHER_SN_FISSION_H_