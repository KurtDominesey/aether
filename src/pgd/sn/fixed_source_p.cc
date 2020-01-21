#include "pgd/sn/fixed_source_p.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::update_last_cache(
      const dealii::BlockVector<double> &mode) {
  caches.back().mode = mode;
  this->d2m.vmult(caches.back().moments, mode);
  this->transport.stream(caches.back().streamed, mode);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products(
      const dealii::BlockVector<double> &mode) {
  std::vector<InnerProducts> inner_products(
      caches.size(), InnerProducts(xs_total.size(), xs_scatter.size()));
  for (int m = 0; m < caches.size(); ++m) {
    dealii::BlockVector<double> mode_m2d(mode);
    this->m2d.vmult(mode_m2d, caches[m].moments);
    // streaming
    for (int n = 0; n < transport.quadrature.size(); ++n)
      inner_products[m].streaming +=
          transport.quadrature.weight(n) *
          (mode.block(n) * caches[m].streamed.block(n));
    // collision and scattering
    std::vector<dealii::types::global_dof_index> dof_indices(
        transport->dof_handler.get_fe().dofs_per_cell);
    using Cell = typename dealii::DoFHandler<dim>::active_cell_iterator;
    int c = 0;
    for (Cell cell = transport->dof_handler.begin_active();
          cell != transport->dof_handler.end(); ++cell, ++c) {
      if (!cell->is_locally_owned())
        continue;
      const dealii::FullMatrix<double> &mass = 
          transport->cell_matrices[c].mass;
      cell->get_dof_indices(dof_indices.size());
      int material = cell->material_id();
      double collision_c = 0;
      double scattering_c = 0;
      for (int n = 0; n < transport.quadrature.size(); ++n) {
        double collision_n = 0;
        double scattering_n = 0;
        for (int i = 0; i < dof_indices.size(); ++i) {
          for (int j = 0; j < dof_indices.size(); ++j) {
            collision_n += mode.block(n)[dof_indices[i]] * mass[i][j] *
                           caches[m].mode.block(n)[dof_indices[j]];
            scattering_n += mode.block(n)[dof_indices[i]] * mass[i][j] *
                            mode_m2d.block(n)[dof_indices[j]];
          }
        }
        collision_c += transport.quadrature.weight(n) * collision_n;
        scattering_c += transport.quadrature.weight(n) * scattering_n;
      }
      inner_products[m].collision[material] += collision_c;
      inner_products[m].scattering[material][0] += scattering_c;
    }
  }
}

}  // namespace aether::pgd::sn