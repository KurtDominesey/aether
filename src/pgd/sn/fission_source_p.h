#ifndef AETHER_PGD_SN_FISSION_SOURCE_P_H_
#define AETHER_PGD_SN_FISSION_SOURCE_P_H

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>

#include "pgd/sn/fixed_source_p.h"
#include "sn/fixed_source.h"
#include "sn/fission.h"
#include "base/petsc_block_block_wrapper.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FissionSourceP : public FixedSourceP<dim, qdim> {
 public:
  std::vector<dealii::BlockVector<double>> zero_sources;
  FissionSourceP(aether::sn::FixedSource<dim, qdim> &fixed_source,
                 aether::sn::Fission<dim, qdim> &fission,
                 Mgxs &mgxs_pseudo, const Mgxs &mgxs);
  double step_eigenvalue(InnerProducts &coefficients);
  // double step(dealii::Vector<double> &delta,
  //             const dealii::BlockVector<double> &b,
  //             std::vector<InnerProducts> coefficients_x,
  //             std::vector<double> coefficients_b,
  //             double omega = 1.0);
 aether::sn::Fission<dim, qdim> &fission;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FISSION_SOURCE_P_H_