#ifndef AETHER_SN_FIXED_SOURCE_PROBLEM_H_
#define AETHER_SN_FIXED_SOURCE_PROBLEM_H_

#include <deal.II/lac/block_vector.h>

#include "base/mgxs.h"
#include "sn/within_group.h"
#include "sn/scattering_block.h"
#include "sn/discrete_to_moment.h"
#include "sn/moment_to_discrete.h"
#include "sn/fixed_source.h"
#include "sn/quadrature.h"

#include "pgd/sn/transport.h"

namespace aether::sn {

/**
 * Convenience class which sets up a fixed-source problem.
 * 
 * This class constructs and stores a fixed-source operator 
 * @ref FixedSourceProblem::fixed_source along with all constituent block
 * operators.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2,
          class TransportType = Transport<dim, qdim>,
          class TransportBlockType = TransportBlock<dim, qdim>>
class FixedSourceProblem {
 protected:
  using BoundaryConditions = 
      std::vector<std::vector<dealii::BlockVector<double>>>;
  //! Vacuum boundary conditions, if needed
  std::unique_ptr<BoundaryConditions> vacuum;

 public:
  /**
   * Constructor.
   * 
   * @param dof_handler DoF handler for the mesh and finite elements.
   * @param quadrature Angular quadrature.
   * @param mgxs Multigroup cross-sections.
   * @param boundary_conditions The values of the incident flux for each
   *                            group, boundary ID, ordinate, and elemental DoF. 
   */
  FixedSourceProblem(const dealii::DoFHandler<dim> &dof_handler,
                     const QAngle<dim, qdim> &quadrature, 
                     const Mgxs &mgxs,
                     const BoundaryConditions &boundary_conditions);

  /**
   * Constructor with vacuum boundaries for boundary_id 0.
   * 
   * @param dof_handler DoF handler for the mesh and finite elements.
   * @param quadrature Angular quadrature.
   * @param mgxs Multigroup cross-sections.
   * @param boundary_conditions The values of the incident flux for each
   *                            group, boundary ID, ordinate, and elemental DoF. 
   */
  FixedSourceProblem(const dealii::DoFHandler<dim> &dof_handler,
                     const QAngle<dim, qdim> &quadrature, 
                     const Mgxs &mgxs);

  /**
   * Apply a transport sweep to the source vector `src`.
   * 
   * @param dst Destination vector \f$\leftarrow\underline{L}^{-1}\f$`src`.
   * @param src Source vector.
   */
  void sweep_source(dealii::BlockVector<double> &dst, 
                    const dealii::BlockVector<double> &src) const;
  //! Fixed-source operator.
  FixedSource<dim, qdim> fixed_source;
  //! Transport operator.
  TransportType transport;
  //! Scattering operator.
  Scattering<dim> scattering;
  //! Discrete-to-moment operator.
  DiscreteToMoment<dim, qdim> d2m;
  //! Moment-to-discrete operator.
  MomentToDiscrete<dim, qdim> m2d;

 protected:
  //! Within-group (diagonal) blocks.
  std::vector<WithinGroup<dim, qdim>> within_groups;
  //! Downscattering (lower triangle) blocks.
  std::vector<std::vector<ScatteringBlock<dim>>> downscattering;
  //! Upscattering (upper triangle) blocks.
  std::vector<std::vector<ScatteringBlock<dim>>> upscattering;
};

template <int dim, int qdim, class TransportType, class TransportBlockType>
FixedSourceProblem<dim, qdim, TransportType, TransportBlockType>::
    FixedSourceProblem(
        const dealii::DoFHandler<dim> &dof_handler,
        const QAngle<dim, qdim> &quadrature, const Mgxs &mgxs,
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
  std::cout << "num_groups: " << num_groups << "\n";
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
  std::cout << "all done\n";
}

template <int dim, int qdim, class TransportType, class TransportBlockType>
FixedSourceProblem<dim, qdim, TransportType, TransportBlockType>::
    FixedSourceProblem(
        const dealii::DoFHandler<dim> &dof_handler,
        const QAngle<dim, qdim> &quadrature, 
        const Mgxs &mgxs) 
    : FixedSourceProblem(dof_handler, quadrature, mgxs, 
        BoundaryConditions(mgxs.total.size(), 
            std::vector<dealii::BlockVector<double>>(1, 
                dealii::BlockVector<double>(
                    quadrature.size(), 
                    dof_handler.get_fe().dofs_per_cell
                )
            )
        )
    ) {}

}  // namespace aether::sn

#endif  // AETHER_SN_FIXED_SOURCE_PROBLEM_H_