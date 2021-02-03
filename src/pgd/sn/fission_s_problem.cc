#include "pgd/sn/fission_s_problem.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FissionSProblem<dim, qdim>::FissionSProblem(
    const dealii::DoFHandler<dim> &dof_handler,
    const aether::sn::QAngle<dim, qdim> &quadrature,
    const Mgxs &mgxs,
    const std::vector<std::vector<dealii::BlockVector<double>>>
      &boundary_conditions,
    const int num_modes)
    : FixedSourceSProblem<dim, qdim>(dof_handler, quadrature, mgxs, boundary_conditions, 
                                     num_modes),
      emission(num_modes),
      production(num_modes),
      fission_s(this->transport, this->m2d, emission, production, this->d2m),
      fission_s_gs(this->transport, this->scattering, this->m2d, this->d2m,
                   this->fixed_source_s.streaming, mgxs, boundary_conditions) {
  const int num_groups = mgxs.total.size();
  const int num_materials = mgxs.total[0].size();
  for (int m = 0; m < num_modes; ++m) {
    for (int mp = 0; mp < num_modes; ++mp) {
      emission[m].emplace_back(dof_handler, this->mgxs_pseudos[m][mp].chi);
      production[m].emplace_back(dof_handler, this->mgxs_pseudos[m][mp].nu_fission);
      for (int g = 0; g < num_groups; ++g) {
        for (int j = 0; j < num_materials; ++j) {
          this->mgxs_pseudos[m][mp].chi[g][j] = this->mgxs.chi[g][j];
        }
      }
    }
  }
}

template <int dim, int qdim>
void FissionSProblem<dim, qdim>::set_cross_sections(
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  FixedSourceSProblem<dim, qdim>::set_cross_sections(coefficients);
  const int num_modes = coefficients.size();
  const int num_groups = this->mgxs.total.size();
  const int num_materials = this->mgxs.total[0].size();
  for (int m = 0; m < num_modes; ++m) {
    for (int mp = 0; mp < num_modes; ++mp) {
      for (int g = 0; g < num_groups; ++g) {
        for (int j = 0; j < num_materials; ++j) {
          this->mgxs_pseudos[m][mp].nu_fission[g][j] = 
              this->mgxs.nu_fission[g][j] * coefficients[m][mp].fission[j];
        }
      }
    }
  }
  fission_s_gs.set_cross_sections(this->mgxs_pseudos);
}

template class FissionSProblem<1>;
template class FissionSProblem<2>;
template class FissionSProblem<3>;

}  // namespace aether::pgd::sn