#ifndef AETHER_PGD_SN_ENERGY_MG_FISS_H_
#define AETHER_PGD_SN_ENERGY_MG_FISS_H_

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>

#include "pgd/sn/energy_mg_full.h"

namespace aether::pgd::sn {

class EnergyMgFiss : public EnergyMgFull {
 public:
  const std::vector<dealii::Vector<double>> zero_sources;
  EnergyMgFiss(const Mgxs &mgxs);
  double step_eigenvalue(InnerProducts &coefficients);
  double update(std::vector<std::vector<InnerProducts>> &coefficients);
  void update(std::vector<std::vector<InnerProducts>> coefficients_x,
              std::vector<std::vector<double>> coefficients_b) override;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_ENERGY_MG_FISS_H_