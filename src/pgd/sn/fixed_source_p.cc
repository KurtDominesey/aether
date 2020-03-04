#include "pgd/sn/fixed_source_p.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FixedSourceP<dim, qdim>::FixedSourceP(
    aether::sn::FixedSource<dim, qdim> &fixed_source,
    Mgxs &mgxs_pseudo, const Mgxs &mgxs,
    std::vector<dealii::BlockVector<double>> &sources)
      : fixed_source(fixed_source), 
        mgxs_pseudo(mgxs_pseudo), mgxs(mgxs),
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
    caches.back().moments.block(g) *= -1;
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
      inner_products[m] = 0;
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
        inner_products[m].collision[material] += mgxs.total[g][material] 
                                                 * collision_c;
        double mgxs_scatter_g_gp = 0;
        for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp)
          mgxs_scatter_g_gp += mgxs.scatter[g][gp][material];
        inner_products[m].scattering[material][0] += mgxs_scatter_g_gp
                                                     * scattering_c;
      }
    }
  }
  std::cout << "ip x " << inner_products[0].streaming << std::endl;
  for (int j = 0; j < inner_products[0].collision.size(); ++j) {
    std::cout << "ip x " << inner_products[0].collision[0] << std::endl;
    std::cout << "ip x " << inner_products[0].scattering[0][0] << std::endl;
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
      source_g = sources[i].block(g);
      transport.collide(collided_g, source_g);
      for (int n = 0; n < transport.quadrature.size(); ++n)
        inner_products[i] += transport.quadrature.weight(n) 
                             * (mode_g.block(n) * collided_g.block(n));
    }
    std::cout << "ip b " << inner_products[i] << std::endl;
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::set_cross_sections(
      InnerProducts &coefficients) {
  AssertDimension(mgxs_pseudo.total.size(), mgxs.total.size());
  for (int g = 0; g < mgxs.total.size(); ++g) {
    AssertDimension(mgxs_pseudo.total[g].size(), mgxs.total[g].size());
    AssertDimension(mgxs_pseudo.scatter[g].size(), mgxs.scatter[g].size());
    for (int material = 0; material < mgxs.total[g].size(); ++material) {
      mgxs_pseudo.total[g][material] = mgxs.total[g][material] *
                                       coefficients.collision[material] /
                                       coefficients.streaming;
      for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
        AssertDimension(mgxs_pseudo.scatter[g][gp].size(), 
                        mgxs.scatter[g][gp].size());
        mgxs_pseudo.scatter[g][gp][material] = 
            mgxs.scatter[g][gp][material] *
            coefficients.scattering[material][0] /
            coefficients.streaming;
      }
    }
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
  subtract_modes_from_source(source, coefficients_x);
  source /= denominator;
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::subtract_modes_from_source(
    dealii::BlockVector<double> &source,
    std::vector<InnerProducts> coefficients_x) {
  AssertDimension(coefficients_x.size(), caches.size() - 1);
  for (int m = 0; m < caches.size() - 1; ++m) {
    for (int g = 0; g < fixed_source.within_groups.size(); ++g) {
      const auto &transport = fixed_source.within_groups[g].transport.transport;
      dealii::BlockVector<double> source_g(transport.quadrature.size(), 
                                           transport.dof_handler.n_dofs());
      dealii::BlockVector<double> mode_g(source_g.get_block_indices());
      dealii::BlockVector<double> mode_m2d_g(source_g.get_block_indices());
      dealii::BlockVector<double> streamed_g(source_g.get_block_indices());
      dealii::BlockVector<double> moments_g(1, transport.dof_handler.n_dofs());
      std::vector<dealii::types::global_dof_index> dof_indices(
          transport.dof_handler.get_fe().dofs_per_cell);
      dealii::Vector<double> streamed_k(dof_indices.size());
      dealii::Vector<double> projected_k(dof_indices.size());
      source_g = source.block(g);
      mode_g = caches[m].mode.block(g);
      streamed_g = caches[m].streamed.block(g);
      moments_g = caches[m].moments.block(g);
      fixed_source.m2d.vmult(mode_m2d_g, moments_g);
      int c = 0;
      for (auto cell = transport.dof_handler.begin_active(); 
           cell != transport.dof_handler.end(); ++cell, ++c) {
        cell->get_dof_indices(dof_indices);
        int material = cell->material_id();
        dealii::FullMatrix<double> mass(transport.cell_matrices[c].mass);
        mass.gauss_jordan();
        for (int n = 0; n < transport.quadrature.size(); ++n) {
          streamed_k = 0;
          for (int i = 0; i < dof_indices.size(); ++i)
            streamed_k[i] = streamed_g.block(n)[dof_indices[i]];
          mass.vmult(projected_k, streamed_k);
          for (int i = 0; i < dof_indices.size(); ++i) {
            source_g.block(n)[dof_indices[i]] +=
                - coefficients_x[m].streaming * projected_k[i]
                - coefficients_x[m].collision[material]
                  * mode_g.block(n)[dof_indices[i]]
                - coefficients_x[m].scattering[material][0]
                  * mode_m2d_g.block(n)[dof_indices[i]];
          }
        }
      }
      source.block(g) = source_g;
    }
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::step(
      dealii::BlockVector<double>&,
      const dealii::BlockVector<double>&,
      std::vector<InnerProducts> coefficients_x,
      std::vector<double> coefficients_b,
      double omega) {
  dealii::BlockVector<double> source(caches.back().mode.get_block_indices());
  dealii::BlockVector<double> uncollided(caches.back().mode.get_block_indices());
  set_cross_sections(coefficients_x.back());
  double denominator = coefficients_x.back().streaming;
  coefficients_x.pop_back();
  get_source(source, coefficients_x, coefficients_b, denominator);
  for (int g = 0; g < fixed_source.within_groups.size(); ++g)
    fixed_source.within_groups[g].transport.vmult(
        uncollided.block(g), source.block(g), false);
  dealii::BlockVector<double> solution(caches.back().mode);
  dealii::SolverControl solver_control(3000, 1e-8);
  dealii::SolverRichardson<dealii::BlockVector<double>> solver(solver_control);
  solver.solve(fixed_source, solution, uncollided, 
               dealii::PreconditionIdentity());
  caches.back().mode.sadd(1 - omega, omega, solution);
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
  set_last_cache();
  get_inner_products_x(inner_products_x);
  get_inner_products_b(inner_products_b);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::normalize() {
  caches.back().mode /= caches.back().mode.l2_norm();
  // throw dealii::ExcNotImplemented();
}

template class FixedSourceP<1>;
template class FixedSourceP<2>;
template class FixedSourceP<3>;

}  // namespace aether::pgd::sn