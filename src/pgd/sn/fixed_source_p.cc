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
void FixedSourceP<dim, qdim>::set_cache(Cache &cache) {
  for (int g = 0; g < fixed_source.within_groups.size(); ++g) {
    fixed_source.d2m.vmult(cache.moments.block(g),
                           cache.mode.block(g));
    const TransportBlock<dim, qdim> &transport =
        dynamic_cast<const TransportBlock<dim, qdim>&>(
            fixed_source.within_groups[g].transport);
    transport.stream(cache.streamed.block(g), 
                     cache.mode.block(g));
    cache.moments.block(g) *= -1;
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::set_last_cache() {
  set_cache(caches.back());
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_x(
    std::vector<InnerProducts> &inner_products, 
    const int m_row, const int m_col_start) {
  for (int m = m_col_start; m < caches.size(); ++m) {
    get_inner_products_x(inner_products[m], caches[m_row].mode, caches[m]);
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_x(
    std::vector<InnerProducts> &inner_products,
    const dealii::Vector<double> &left) {
  dealii::BlockVector<double> left_b(caches.back().mode);
  left_b = left;
  for (int m = 0; m < caches.size(); ++m) {
    get_inner_products_x(inner_products[m], left_b, caches[m]);
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_x(
    InnerProducts &inner_products, 
    const dealii::Vector<double> &left, 
    const dealii::Vector<double> &right) {
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(
          fixed_source.within_groups[0].transport.transport);
  Cache cache(fixed_source.within_groups.size(), transport.quadrature.size(), 1,
              transport.dof_handler.n_dofs());
  cache.mode = right;
  set_cache(cache);
  dealii::BlockVector<double> left_b(cache.mode);
  left_b = left;
  get_inner_products_x(inner_products, left_b, cache);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_x(
    InnerProducts &inner_products,
    const dealii::Vector<double> &right) {
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(
          fixed_source.within_groups[0].transport.transport);
  Cache cache(fixed_source.within_groups.size(), transport.quadrature.size(), 1,
              transport.dof_handler.n_dofs());
  cache.mode = right;
  set_cache(cache);
  get_inner_products_x(inner_products, caches.back().mode, cache);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_x(
    InnerProducts &inner_products, 
    const dealii::BlockVector<double> &left,
    const Cache &cache) {
  for (int g = 0; g < fixed_source.within_groups.size(); ++g) {
    const Transport<dim, qdim> &transport =
        dynamic_cast<const Transport<dim, qdim>&>(
            fixed_source.within_groups[g].transport.transport);
    dealii::BlockVector<double> mode_last_g(transport.quadrature.size(),
                                            transport.dof_handler.n_dofs());
    mode_last_g = left.block(g);
    inner_products = 0;
    dealii::BlockVector<double> mode_g(mode_last_g.get_block_indices());
    dealii::BlockVector<double> mode_m2d_g(mode_last_g.get_block_indices());
    dealii::BlockVector<double> streamed_g(mode_last_g.get_block_indices());
    dealii::BlockVector<double> moments_g(1, transport.dof_handler.n_dofs());
    mode_g = cache.mode.block(g);
    streamed_g = cache.streamed.block(g);
    moments_g = cache.moments.block(g);
    fixed_source.m2d.vmult(mode_m2d_g, moments_g);
    // streaming
    for (int n = 0; n < transport.quadrature.size(); ++n)
      inner_products.streaming +=
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
      inner_products.collision[material] += mgxs.total[g][material] 
                                                * collision_c;
      double mgxs_scatter_g_gp = 0;
      for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp)
        mgxs_scatter_g_gp += mgxs.scatter[g][gp][material];
      inner_products.scattering[material][0] += mgxs_scatter_g_gp
                                                    * scattering_c;
    }
  }
  for (int j = 0; j < inner_products.scattering.size(); ++j) {
    inner_products.fission[j] = inner_products.scattering[j][0];
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_b(
    std::vector<double> &inner_products, const int m_row) {
  get_inner_products_b(inner_products, caches[m_row].mode);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_b(
    std::vector<double> &inner_products, const dealii::Vector<double> &left) {
  dealii::BlockVector<double> left_b(caches.back().mode);
  left_b = left;
  get_inner_products_b(inner_products, left_b);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products_b(
    std::vector<double> &inner_products, 
    const dealii::BlockVector<double> &left) {
  for (int i = 0; i < sources.size(); ++i) {
    inner_products[i] = 0;
    for (int g = 0; g < fixed_source.within_groups.size(); ++g) {
      const Transport<dim, qdim> &transport =
          dynamic_cast<const Transport<dim, qdim>&>(
              fixed_source.within_groups[g].transport.transport);
      dealii::BlockVector<double> mode_g(transport.get_block_indices());
      dealii::BlockVector<double> source_g(transport.get_block_indices());
      dealii::BlockVector<double> collided_g(transport.get_block_indices());
      mode_g = left.block(g);
      source_g = sources[i].block(g);
      transport.collide(collided_g, source_g);
      for (int n = 0; n < transport.quadrature.size(); ++n)
        inner_products[i] += transport.quadrature.weight(n) 
                             * (mode_g.block(n) * collided_g.block(n));
    }
  }
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::set_cross_sections(
      const InnerProducts &coefficients) {
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
    const std::vector<InnerProducts> &coefficients_x,
    const std::vector<double> &coefficients_b,
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
double FixedSourceP<dim, qdim>::step(
      dealii::Vector<double> &delta,
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
  // fixed_source.vmult(solution, caches.back().mode);
  // uncollided -= solution;
  // caches.back().mode += uncollided;
  dealii::IterationNumberControl solver_control(25, 1e-8);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver(solver_control,
      dealii::SolverGMRES<dealii::BlockVector<double>>::AdditionalData(32));
  solver.solve(fixed_source, solution, uncollided, 
               dealii::PreconditionIdentity());
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(
          fixed_source.within_groups[0].transport.transport);
  double norm = transport.norm(solution.block(0));
  // solution /= norm;
  // std::cout << "norm " << norm << std::endl;
  dealii::Vector<double> mode;
  mode = caches.back().mode;
  delta = solution;
  delta -= mode;
  return mode.l2_norm();
  // caches.back().mode.sadd(1 - omega, omega, solution);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::take_step(
    const double factor, const dealii::Vector<double> &delta) {
  dealii::BlockVector<double> delta_b(caches.back().mode.get_block_indices());
  delta_b = delta;
  caches.back().mode.add(factor, delta_b);
}

template <int dim, int qdim>
double FixedSourceP<dim, qdim>::get_residual(
      std::vector<InnerProducts> coefficients_x,
      std::vector<double> coefficients_b) {
  dealii::BlockVector<double> source(caches.back().mode.get_block_indices());
  dealii::BlockVector<double> uncollided(caches.back().mode.get_block_indices());
  set_cross_sections(coefficients_x.back());
  double denominator = coefficients_x.back().streaming;
  coefficients_x.pop_back();
  get_source(source, coefficients_x, coefficients_b, denominator);
  for (int g = 0; g < fixed_source.within_groups.size(); ++g)
    fixed_source.within_groups[g].transport.vmult(
        uncollided.block(g), source.block(g), false);
  dealii::BlockVector<double> operated(caches.back().mode);
  fixed_source.vmult(operated, caches.back().mode);
  dealii::BlockVector<double> residual(uncollided);
  residual -= operated;
  return residual.l2_norm() / uncollided.l2_norm();
}

template <int dim, int qdim>
double FixedSourceP<dim, qdim>::line_search(
    std::vector<double> &c,
    const dealii::Vector<double> &step,
    const InnerProducts &coefficients_x_mode_step,
    const InnerProducts &coefficients_x_step_step,
    std::vector<InnerProducts> coefficients_x_mode_mode,
    std::vector<InnerProducts> coefficients_x_step_mode,
    const std::vector<double> &coefficients_b_mode,
    const std::vector<double> &coefficients_b_step) {
  dealii::BlockVector<double> source(caches.back().mode.get_block_indices());
  dealii::BlockVector<double> uncollided(caches.back().mode.get_block_indices());
  dealii::BlockVector<double> operated(caches.back().mode.get_block_indices());
  dealii::BlockVector<double> v0(operated);
  dealii::BlockVector<double> v1(operated);
  dealii::BlockVector<double> v2(operated);
  dealii::BlockVector<double> v3(operated);
  dealii::BlockVector<double> step_b(operated);
  step_b = step;
  // (mode, mode)
  set_cross_sections(coefficients_x_mode_mode.back());
  double factor_mm = coefficients_x_mode_mode.back().streaming;
  coefficients_x_mode_mode.pop_back();
  get_source(source, coefficients_x_mode_mode, coefficients_b_mode, 1);
  for (int g = 0; g < fixed_source.within_groups.size(); ++g)
    fixed_source.within_groups[g].transport.vmult(
        uncollided.block(g), source.block(g), false);
  // dealii::BlockVector<double> f0(uncollided);
  v0 -= uncollided;
  fixed_source.vmult(operated, caches.back().mode);
  v0.add(factor_mm, operated);
  fixed_source.vmult(operated, step_b);
  v1.add(factor_mm, operated);
  // (step, mode)
  set_cross_sections(coefficients_x_step_mode.back());
  double factor_sm = coefficients_x_step_mode.back().streaming;
  coefficients_x_step_mode.pop_back();
  get_source(source, coefficients_x_step_mode, coefficients_b_step, 1);
  for (int g = 0; g < fixed_source.within_groups.size(); ++g)
    fixed_source.within_groups[g].transport.vmult(
        uncollided.block(g), source.block(g), false);
  // dealii::BlockVector<double> f1(uncollided);
  v1 -= uncollided;
  fixed_source.vmult(operated, caches.back().mode);
  v1.add(factor_sm, operated);
  fixed_source.vmult(operated, step_b);
  v2.add(factor_sm, operated);
  // (mode, step)
  set_cross_sections(coefficients_x_mode_step);
  double factor_ms = coefficients_x_mode_step.streaming;
  fixed_source.vmult(operated, caches.back().mode);
  v1.add(factor_ms, operated);
  fixed_source.vmult(operated, step_b);
  v2.add(factor_ms, operated);
  // (step, step)
  set_cross_sections(coefficients_x_step_step);
  double factor_ss = coefficients_x_step_step.streaming;
  fixed_source.vmult(operated, caches.back().mode);
  v2.add(factor_ss, operated);
  fixed_source.vmult(operated, step_b);
  v3.add(factor_ss, operated);
  // get coefficients
  c[0] += v0 * v0;
  c[1] += 2 * (v1 * v0);
  c[2] += 2 * (v2 * v0) + v1 * v1;
  c[3] += 2 * (v2 * v1);
  c[4] += 2 * (v3 * v1) + v2 * v2;
  c[5] += 2 * (v3 * v2);
  c[6] += v3 * v3;
  return 1;
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                    const dealii::BlockVector<double> &src,
                                    std::vector<InnerProducts> coefficients_x,
                                    std::vector<double> coefficients_b) {

}

template <int dim, int qdim>
double FixedSourceP<dim, qdim>::enrich(const double factor) {
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(
          fixed_source.within_groups[0].transport.transport);
  caches.emplace_back(fixed_source.within_groups.size(),
                      transport.quadrature.size(), 
                      1,
                      transport.dof_handler.n_dofs());
  caches.back().mode = 1;
  return 0;
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products(
    std::vector<InnerProducts> &inner_products_x, 
    std::vector<double> &inner_products_b) {
  set_last_cache();
  const int m_row = caches.size() - 1;
  const int m_col_start = 0;
  get_inner_products_x(inner_products_x, m_row, m_col_start);
  get_inner_products_b(inner_products_b, m_row);
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::get_inner_products(
    std::vector<InnerProducts> &inner_products_x, 
    std::vector<double> &inner_products_b,
    const int m_row, const int m_col_start) {
  get_inner_products_x(inner_products_x, m_row, m_col_start);
  get_inner_products_b(inner_products_b, m_row);
}

template <int dim, int qdim>
double FixedSourceP<dim, qdim>::normalize() {
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(
          fixed_source.within_groups[0].transport.transport);
  // return 0;
  dealii::Vector<double> &mode = caches.back().mode.block(0);
  // dealii::Vector<double> one(mode);
  // one = 1;
  // dealii::Vector<double> mode_l2(mode);
  // transport.collide(mode_l2, mode);
  // mode /= std::sqrt(mode * mode_l2);
  // for (int i = 0; i < mode.size(); ++i)
  //   mode[i] *= mode[i] > 0 ? 1 : -1;
  // double denom = std::pow(transport.inner_product(one, mode), 2);
  double denom = transport.norm(mode);
  mode /= denom;
  return denom;
  // throw dealii::ExcNotImplemented();
}

template <int dim, int qdim>
void FixedSourceP<dim, qdim>::scale(double factor) {
  throw dealii::ExcNotImplemented();
}

template class FixedSourceP<1>;
template class FixedSourceP<2>;
template class FixedSourceP<3>;

}  // namespace aether::pgd::sn