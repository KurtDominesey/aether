#include "pgd/sn/fixed_source_s_problem.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FixedSourceSProblem<dim, qdim>::FixedSourceSProblem(
    const dealii::DoFHandler<dim> &dof_handler,
    const aether::sn::QAngle<dim, qdim> &quadrature,
    const Mgxs &mgxs,
    const std::vector<std::vector<dealii::BlockVector<double>>>
      &boundary_conditions,
    const int num_modes)
    : transport(dof_handler, quadrature),
      scattering(dof_handler),
      m2d(quadrature),
      d2m(quadrature),
      mgxs(mgxs),
      within_groups(num_modes,
        std::vector<WithinGroups>(num_modes)),
      downscattering(num_modes,
        std::vector<ScatteringTriangle>(num_modes,
          ScatteringTriangle(mgxs.total.size()))),
      upscattering(num_modes,
        std::vector<ScatteringTriangle>(num_modes,
          ScatteringTriangle(mgxs.total.size()))),
      mgxs_pseudos(num_modes,
        std::vector<Mgxs>(num_modes, mgxs)),
      blocks(num_modes),
      fixed_source_s(blocks, m2d, d2m, mgxs),
      fixed_source_s_gs(transport, scattering, m2d, d2m, 
                        fixed_source_s.streaming, mgxs, boundary_conditions) {
  const int num_groups = mgxs.total.size();
  for (int m = 0; m < num_modes; ++m) {
    for (int mp = 0; mp < num_modes; ++mp) {
      for (int g = 0; g < num_groups; ++g) {
        auto transport_wg = std::make_shared<TransportBlock<dim, qdim>>(
            transport, mgxs_pseudos[m][mp].total[g], boundary_conditions[g]);
        auto scattering_wg = std::make_shared<aether::sn::ScatteringBlock<dim>>(
            scattering, mgxs_pseudos[m][mp].scatter[g][g]);
        within_groups[m][mp].emplace_back(transport_wg, m2d, scattering_wg, d2m);
        for (int gp = g - 1; gp >= 0; --gp)  // from g' to g
          downscattering[m][mp][g].emplace_back(
              scattering, mgxs_pseudos[m][mp].scatter[g][gp]);
        for (int gp = g + 1; gp < num_groups; ++gp)  // from g' to g
          upscattering[m][mp][g].emplace_back(
              scattering, mgxs_pseudos[m][mp].scatter[g][gp]);
      }
      blocks[m].emplace_back(within_groups[m][mp], downscattering[m][mp], 
                             upscattering[m][mp], m2d, d2m);
    }
  }
}

template <int dim, int qdim>
void FixedSourceSProblem<dim, qdim>::set_cross_sections(
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  const int num_modes = coefficients.size();
  const int num_groups = mgxs.total.size();
  const int num_materials = mgxs.total[0].size();
  for (int m = 0; m < num_modes; ++m) {
    AssertDimension(coefficients[m].size(), num_modes);
    for (int mp = 0; mp < num_modes; ++mp) {
      Mgxs &mgxs_pseudo = mgxs_pseudos[m][mp];
      fixed_source_s.streaming[m][mp] = coefficients[m][mp].streaming;
      for (int g = 0; g < num_groups; ++g) {
        for (int j = 0; j < num_materials; ++j) {
          mgxs_pseudo.total[g][j] = 
              mgxs.total[g][j] * coefficients[m][mp].collision[j];
          for (int gp = 0; gp < num_groups; ++gp) {
            mgxs_pseudo.scatter[g][gp][j] = 
                mgxs.scatter[g][gp][j] * coefficients[m][mp].scattering[j][0];
          }
        }
      }
    }
  }
  fixed_source_s_gs.set_cross_sections(mgxs_pseudos);
}

template <int dim, int qdim>
void FixedSourceSProblem<dim, qdim>::sweep_source(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src) const {
  const int num_modes = blocks.size();
  const int num_groups = blocks[0][0].within_groups.size();
  for (int m = 0, mg = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g, ++mg) {
      blocks[m][m].within_groups[g].transport.vmult(
          dst.block(mg), src.block(mg), false);
    }
  }
}

template <int dim, int qdim>
double FixedSourceSProblem<dim, qdim>::l2_norm(
    const dealii::BlockVector<double> &modes) const {
  double square = 0;
  for (int b = 0; b < modes.n_blocks(); ++b)
    square += transport.inner_product(modes.block(b), modes.block(b));
  return std::sqrt(square);
}

template class FixedSourceSProblem<1>;
template class FixedSourceSProblem<2>;
template class FixedSourceSProblem<3>;

}  // namespace aether::pgd::sn