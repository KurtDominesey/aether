#include "pgd/sn/fixed_source_p.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FixedSourceP<dim, qdim>::FixedSourceP(
    aether::sn::FixedSource<dim, qdim> &fixed_source,
    std::vector<double> &cross_sections_total_w,
    std::vector<std::vector<double>> &cross_sections_scatter_w,
    const std::vector<double> &cross_sections_total_r,
    const std::vector<std::vector<double>> &cross_sections_scatter_r,
    std::vector<dealii::BlockVector<double>> &sources)
      : fixed_source(fixed_source),
        cross_sections_total_w(cross_sections_total_w),
        cross_sections_scatter_w(cross_sections_scatter_w),
        cross_sections_total_r(cross_sections_total_r),
        cross_sections_scatter_r(cross_sections_scatter_r),
        sources(sources) {}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::set_last_cache() {
  for (int g = 0; g < fixed_source.within_groups.size(); ++g) {
    fixed_source.d2m.vmult(caches.back().moments.block(g),
                           caches.back().mode.block(g));
    const TransportBlock<dim, qdim> &transport =
        dynamic_cast<const TransportBlock<dim, qdim>&>(
            fixed_source.within_groups[g].transport);
    transport.stream(caches.back().streamed.block(g), 
                     caches.back().mode.block(g));
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_x(
    std::vector<InnerProducts> &inner_products) {
  for (int g = 0; g < fixed_source.within_groups.size(); ++g) {
    const Transport<dim, qdim> &transport =
        dynamic_cast<const Transport<dim, qdim>&>(
            fixed_source.within_groups[g].transport.transport);
    dealii::BlockVector<double> mode_last_g(transport.quadrature.size(),
                                            transport.dof_handler.n_dofs());
    mode_last_g = caches.back().mode.block(g);
    for (int m = 0; m < caches.size(); ++m) {
      inner_products[m].streaming = 0;
      inner_products[m].collision = 0;
      for (int mat = 0; mat < inner_products[m].scattering.size(); ++mat)
        inner_products[m].scattering[mat] = 0;
      dealii::BlockVector<double> mode_g(mode_last_g.get_block_indices());
      dealii::BlockVector<double> mode_m2d_g(mode_last_g.get_block_indices());
      dealii::BlockVector<double> streamed_g(mode_last_g.get_block_indices());
      dealii::BlockVector<double> moments_g(1, transport.dof_handler.n_dofs());
      mode_g = caches[m].mode.block(g);
      streamed_g = caches[m].streamed.block(g);
      moments_g = caches[m].moments.block(g);
      fixed_source.m2d.vmult(mode_m2d_g, moments_g);
      // streaming
      for (int n = 0; n < transport.quadrature.size(); ++n)
        inner_products[m].streaming +=
            transport.quadrature.weight(n) *
            (mode_last_g.block(n) * streamed_g.block(n));
      // collision and scattering
      std::vector<dealii::types::global_dof_index> dof_indices(
          transport.dof_handler.get_fe().dofs_per_cell);
      using Cell = typename dealii::DoFHandler<dim>::active_cell_iterator;
      int c = 0;
      for (Cell cell = transport.dof_handler.begin_active();
            cell != transport.dof_handler.end(); ++cell, ++c) {
        if (!cell->is_locally_owned())
          continue;
        const dealii::FullMatrix<double> &mass = 
            transport.cell_matrices[c].mass;
        cell->get_dof_indices(dof_indices);
        int material = cell->material_id();
        double collision_c = 0;
        double scattering_c = 0;
        for (int n = 0; n < transport.quadrature.size(); ++n) {
          double collision_n = 0;
          double scattering_n = 0;
          for (int i = 0; i < dof_indices.size(); ++i) {
            for (int j = 0; j < dof_indices.size(); ++j) {
              collision_n += mode_last_g.block(n)[dof_indices[i]] 
                             * mass[i][j] * mode_g.block(n)[dof_indices[j]];
              scattering_n += mode_last_g.block(n)[dof_indices[i]] 
                              * mass[i][j] * mode_m2d_g.block(n)[dof_indices[j]];
            }
          }
          collision_c += transport.quadrature.weight(n) * collision_n;
          scattering_c += transport.quadrature.weight(n) * scattering_n;
        }
        inner_products[m].collision[material] +=
            cross_sections_total_r[material] * collision_c;
        inner_products[m].scattering[material][0] += 
            cross_sections_scatter_r[material][0] * scattering_c;
      }
    }
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_b(
    std::vector<double> &inner_products) {
  for (int i = 0; i < sources.size(); ++i) {
    inner_products[i] = 0;
    for (int g = 0; g < fixed_source.within_groups.size(); ++g) {
      const Transport<dim, qdim> &transport =
          dynamic_cast<const Transport<dim, qdim>&>(
              fixed_source.within_groups[g].transport.transport);
      dealii::BlockVector<double> mode_g(transport.get_block_indices());
      dealii::BlockVector<double> source_g(transport.get_block_indices());
      dealii::BlockVector<double> collided_g(transport.get_block_indices());
      mode_g = caches.back().mode.block(g);
      transport.collide(collided_g, source_g);
      for (int n = 0; n < transport.quadrature.size(); ++n)
        inner_products[i] +=
            transport.quadrature.weight(n) * (mode_g * collided_g);
    }
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::set_cross_sections(
      InnerProducts &coefficients) {
  AssertDimension(cross_sections_total_w.size(), cross_sections_total_r.size());
  for (int material = 0; material < cross_sections_total_w.size(); ++material) {
    cross_sections_total_w[material] = cross_sections_total_r[material] *
                                       coefficients.collision[material] /
                                       coefficients.streaming;
    cross_sections_scatter_w[material][0] =
        cross_sections_scatter_r[material][0] *
        coefficients.scattering[material][0] / coefficients.streaming;
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_source(
    dealii::BlockVector<double> &source,
    std::vector<InnerProducts> &coefficients_x,
    std::vector<double> &coefficients_b,
    double denominator) {
  source = 0;
  for (int i = 0; i < coefficients_b.size(); ++i)
    source.add(coefficients_b[i], sources[i]);
  // subtract_modes_from_source(source, coefficients_x);
  source /= denominator;
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::subtract_modes_from_source(
    dealii::BlockVector<double> &source,
    std::vector<InnerProducts> coefficients_x) {
  for (int m = 0; m < caches.size() - 1; ++m) {

  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::step(
      dealii::BlockVector<double>&,
      const dealii::BlockVector<double>&,
      std::vector<InnerProducts> coefficients_x,
      std::vector<double> coefficients_b) {
  dealii::BlockVector<double> source(caches.back().mode.get_block_indices());
  dealii::BlockVector<double> uncollided(caches.back().mode.get_block_indices());
  set_cross_sections(coefficients_x.back());
  double denominator = coefficients_x.back().streaming;
  coefficients_x.pop_back();
  get_source(source, coefficients_x, coefficients_b, denominator);
  for (int g = 0; g < fixed_source.within_groups.size(); ++g)
    fixed_source.within_groups[0].transport.vmult(uncollided.block(g), 
                                                  source.block(g), false);
  dealii::ReductionControl solver_control(500, 1e-8, 1e-8);
  dealii::SolverRichardson<dealii::BlockVector<double>> solver(solver_control);
  solver.solve(fixed_source, caches.back().mode, uncollided, 
               dealii::PreconditionIdentity());
  set_last_cache();
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                    const dealii::BlockVector<double> &src,
                                    std::vector<InnerProducts> coefficients_x,
                                    std::vector<double> coefficients_b) {

}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::enrich() {
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(
          fixed_source.within_groups[0].transport.transport);
  caches.emplace_back(fixed_source.within_groups.size(),
                      transport.quadrature.size(), 
                      1,
                      transport.dof_handler.n_dofs());
  caches.back().mode = 1;
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products(
    std::vector<InnerProducts> &inner_products_x, 
    std::vector<double> &inner_products_b) {
  get_inner_products_x(inner_products_x);
  get_inner_products_b(inner_products_b);
}

template class FixedSourceP<1>;
template class FixedSourceP<2>;
template class FixedSourceP<3>;

}  // namespace aether::pgd::sn