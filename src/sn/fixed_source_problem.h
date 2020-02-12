#ifndef AETHER_SN_FIXED_SOURCE_PROBLEM_H_
#define AETHER_SN_FIXED_SOURCE_PROBLEM_H_

#include <deal.II/lac/block_vector.h>

#include "base/mgxs.h"
#include "sn/within_group.h"
#include "sn/scattering_block.h"
#include "sn/discrete_to_moment.h"
#include "sn/moment_to_discrete.h"
#include "sn/fixed_source.h"

namespace aether::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2,
          class TransportType = Transport<dim, qdim>,
          class TransportBlockType = TransportBlock<dim, qdim>>
class FixedSourceProblem {
 public:
  FixedSourceProblem(const dealii::DoFHandler<dim> &dof_handler,
                     const dealii::Quadrature<qdim> &quadrature, 
                     const Mgxs &mgxs,
                     const std::vector<std::vector<dealii::BlockVector<double>>>
                         &boundary_conditions);
  void sweep_source(dealii::BlockVector<double> &dst, 
                    const dealii::BlockVector<double> &src) const;
  FixedSource<dim, qdim> fixed_source;

 protected:
  TransportType transport;
  Scattering<dim> scattering;
  DiscreteToMoment<qdim> d2m;
  MomentToDiscrete<qdim> m2d;
  std::vector<WithinGroup<dim, qdim>> within_groups;
  std::vector<std::vector<ScatteringBlock<dim>>> downscattering;
  std::vector<std::vector<ScatteringBlock<dim>>> upscattering;
};

template <int dim, int qdim, class TransportType, class TransportBlockType>
FixedSourceProblem<dim, qdim, TransportType, TransportBlockType>::
    FixedSourceProblem(
        const dealii::DoFHandler<dim> &dof_handler,
        const dealii::Quadrature<qdim> &quadrature, const Mgxs &mgxs,
        const std::vector<std::vector<dealii::BlockVector<double>>>
            &boundary_conditions)
    : transport(dof_handler, quadrature),
      scattering(dof_handler),
      d2m(quadrature),
      m2d(quadrature),
      fixed_source(within_groups, downscattering, upscattering, m2d, d2m) {
  const int num_groups = mgxs.total.size();
  AssertDimension(num_groups, mgxs.scatter.size());
  AssertDimension(num_groups, boundary_conditions.size());
  downscattering.resize(num_groups);
  upscattering.resize(num_groups);
  for (int g = 0; g < num_groups; ++g) {
    AssertDimension(mgxs.scatter[g].size(), num_groups);
    auto transport_wg = std::make_shared<TransportBlockType>(
        transport, mgxs.total[g], boundary_conditions[g]);
    auto scattering_wg = std::make_shared<ScatteringBlock<dim>>(
        scattering, mgxs.scatter[g][g]);
    within_groups.emplace_back(transport_wg, m2d, scattering_wg, d2m);
    for (int gp = g - 1; gp >= 0; --gp)  // from g' to g
      downscattering[g].emplace_back(scattering, mgxs.scatter[g][gp]);
    for (int gp = g + 1; gp < num_groups; ++gp)  // from g' to g
      upscattering[g].emplace_back(scattering, mgxs.scatter[g][gp]);
  }
}

}  // namespace aether::sn

#endif  // AETHER_SN_FIXED_SOURCE_PROBLEM_H_