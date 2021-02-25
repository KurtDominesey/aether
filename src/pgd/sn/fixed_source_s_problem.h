#ifndef AETHER_PGD_SN_FIXED_SOURCE_S_PROBLEM_H_
#define AETHER_PGD_SN_FIXED_SOURCE_S_PROBLEM_H_

#include <deal.II/lac/block_vector.h>

#include "base/mgxs.h"
#include "sn/within_group.h"
#include "sn/scattering_block.h"
#include "sn/discrete_to_moment.h"
#include "sn/moment_to_discrete.h"
#include "sn/fixed_source.h"
#include "sn/quadrature.h"

#include "pgd/sn/transport.h"
#include "pgd/sn/transport_block.h"
#include "pgd/sn/fixed_source_s.h"
#include "pgd/sn/fixed_source_s_gs.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSourceSProblem {
 public:
  /**
   * Constructor.
   */
  FixedSourceSProblem(
      const dealii::DoFHandler<dim> &dof_handler,
      const aether::sn::QAngle<dim, qdim> &quadrature,
      const Mgxs &mgxs,
      const std::vector<std::vector<dealii::BlockVector<double>>>
        &boundary_conditions,
      const int num_modes);

  /**
   * Set cross-sections.
   */
  void set_cross_sections(
      const std::vector<std::vector<InnerProducts>> &coefficients);

  /**
   * Sweep source.
   */
  void sweep_source(dealii::BlockVector<double> &dst, 
                    const dealii::BlockVector<double> &src) const;

  /**
   * L2 norm of modes.
   */
  double l2_norm(const dealii::BlockVector<double> &modes) const;

 protected:
  Transport<dim, qdim> transport;
  aether::sn::Scattering<dim> scattering;
  aether::sn::MomentToDiscrete<dim, qdim> m2d;
  aether::sn::DiscreteToMoment<dim, qdim> d2m;
  const Mgxs &mgxs;
  using WithinGroups = std::vector<aether::sn::WithinGroup<dim, qdim>>;
  std::vector<std::vector<WithinGroups>> within_groups;
  using ScatteringTriangle
      = std::vector<std::vector<aether::sn::ScatteringBlock<dim>>>;
  std::vector<std::vector<ScatteringTriangle>> upscattering;
  std::vector<std::vector<ScatteringTriangle>> downscattering;
  std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> blocks;
  std::vector<std::vector<Mgxs>> mgxs_pseudos;

 public:
  FixedSourceS<dim, qdim> fixed_source_s;
  FixedSourceSGS<dim, qdim> fixed_source_s_gs;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_S_PROBLEM_H_