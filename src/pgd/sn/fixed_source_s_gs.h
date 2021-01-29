#ifndef AETHER_PGD_SN_FIXED_SOURCE_S_GS_H_
#define AETHER_PGD_SN_FIXED_SOURCE_S_GS_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>

#include "base/mgxs.h"
#include "sn/within_group.h"
#include "sn/scattering_block.h"
#include "sn/discrete_to_moment.h"
#include "sn/moment_to_discrete.h"
#include "sn/fixed_source.h"

#include "pgd/sn/transport.h"
#include "pgd/sn/transport_block.h"
#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSourceSGS {
 public:
  FixedSourceSGS(
      const Transport<dim, qdim> &transport,
      const aether::sn::Scattering<dim> &scattering,
      const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
      const aether::sn::DiscreteToMoment<dim, qdim> &d2m,
      const std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> &blocks,
      const std::vector<std::vector<double>> &streaming,
      const Mgxs &mgxs,
      const std::vector<std::vector<dealii::BlockVector<double>>>
        &boundary_conditions);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  void set_cross_sections(const std::vector<std::vector<Mgxs>> &mgxs);
 protected:
  //! Fixed-source blocks
  const std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> &blocks;
  //! Moment to discrete operator, \f$M\f$
  const aether::sn::MomentToDiscrete<dim, qdim> &m2d;
  //! Discrete to moment operator, \f$D\f$
  const aether::sn::DiscreteToMoment<dim, qdim> &d2m;
  //! Streaming coefficients
  const std::vector<std::vector<double>> &streaming;
  //! Diagonal blocks
  std::vector<std::vector<aether::sn::WithinGroup<dim, qdim>>> within_groups;
  //! Diagonal cross-sections
  std::vector<Mgxs> mgxs_wg;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_S_GS_H_