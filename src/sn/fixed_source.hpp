#ifndef AETHER_SN_FIXED_SOURCE_H_
#define AETHER_SN_FIXED_SOURCE_H_

#include "within_group.hpp"
#include "scattering.hpp"

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSource {
 public:
  FixedSource(std::vector<WithinGroup<dim, qdim>> &within_groups,
              std::vector<std::vector<Scattering<dim>>> &downscattering,
              std::vector<std::vector<Scattering<dim>>> &upscattering,
              MomentToDiscrete<qdim> &m2d,
              DiscreteToMoment<qdim> &d2m);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  
 protected:
  const std::vector<WithinGroup<dim, qdim>> &within_groups;
  const std::vector<std::vector<Scattering<dim>>> &downscattering;
  const std::vector<std::vector<Scattering<dim>>> &upscattering;
  const MomentToDiscrete<qdim> &m2d;
  const DiscreteToMoment<qdim> &d2m;
};

#endif  // AETHER_SN_FIXED_SOURCE_H_