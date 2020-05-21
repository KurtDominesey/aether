#ifndef AETHER_SN_FIXED_SOURCE_H_
#define AETHER_SN_FIXED_SOURCE_H_

#include "within_group.h"
#include "scattering_block.h"
#include "discrete_to_moment.h"
#include "moment_to_discrete.h"

namespace aether::pgd::sn {
template <int dim, int qdim> class FixedSourceP;
}

namespace aether::sn {

template <class SolverType, int dim, int qdim> class FixedSourceGS;

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSource {
 public:
  FixedSource(std::vector<WithinGroup<dim, qdim>> &within_groups,
              std::vector<std::vector<ScatteringBlock<dim>>> &downscattering,
              std::vector<std::vector<ScatteringBlock<dim>>> &upscattering,
              MomentToDiscrete<dim, qdim> &m2d,
              DiscreteToMoment<dim, qdim> &d2m);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  
 protected:
  const std::vector<WithinGroup<dim, qdim>> &within_groups;
  const std::vector<std::vector<ScatteringBlock<dim>>> &downscattering;
  const std::vector<std::vector<ScatteringBlock<dim>>> &upscattering;
  const MomentToDiscrete<dim, qdim> &m2d;
  const DiscreteToMoment<dim, qdim> &d2m;
  friend class aether::pgd::sn::FixedSourceP<dim, qdim>;
  template <class SolverType, int dimm, int qdimm>
  friend class FixedSourceGS;
};

}  // namespace aether::sn

#endif  // AETHER_SN_FIXED_SOURCE_H_