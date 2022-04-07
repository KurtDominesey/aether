#include "pgd/sn/fixed_source_2D1D.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FixedSource2D1D<dim, qdim>::FixedSource2D1D(
    const aether::sn::FixedSource<dim, qdim> &fixed_src,
    const std::vector<dealii::BlockVector<double>> &srcs,
    const Transport<dim, qdim> &transport,
    Mgxs &mgxs_rom) 
    : fixed_src(fixed_src), 
      srcs(srcs),
      transport(transport),
      dof_zones(transport.dof_handler.n_dofs()), 
      mgxs_rom(mgxs_rom),
      control_wg(100, 1e-3, 1e-7),
      solver_wg(control_wg, SolverWG::AdditionalData(32)),
      control(100, 1e-2, 1e-6),
      solver(control, Solver::AdditionalData(32)),
      fixed_src_gs(fixed_src, solver_wg) {
  scaling.reinit(transport.dof_handler.n_dofs());
  src.reinit(mgxs_rom.num_groups, 
      transport.quadrature.size()*transport.dof_handler.n_dofs());
  uncollided.reinit(src);
  iprods_src.resize(srcs.size(), std::vector<double>(mgxs_rom.num_groups));
  std::vector<dealii::types::global_dof_index> dof_indices(
      transport.dof_handler.get_fe().dofs_per_cell);
  for (const auto &cell : transport.dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;
    cell->get_dof_indices(dof_indices);
    for (dealii::types::global_dof_index i : dof_indices)
      dof_zones[i] = cell->material_id();
  }
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::enrich() {
  Products prod = Products(
      mgxs_rom.num_groups, transport.quadrature.size(), dof_zones.size());
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    for (int n = 0; n < transport.quadrature.size(); ++n) {
      int nn = n * dof_zones.size();
      for (int i = 0; i < dof_zones.size(); ++i) {
        prod.psi.block(g)[nn+i] = !dof_zones[i];
      }
    }
  }
  prod.psi = 1;
  prod.test = prod.psi;
  set_products(prod);
  prods.push_back(prod);
  iprods_flux.emplace_back(mgxs_rom.num_groups, mgxs_rom.num_materials);
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::normalize(bool groupwise, bool discrete) {
  // double norm = 0;  //prods.back().psi.l2_norm();
  std::vector<double> norms_trial(mgxs_rom.num_groups);
  std::vector<double> norms_test(mgxs_rom.num_groups);
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    if (discrete) {  // l2 norm (of a vector)
      norms_trial[g] = prods.back().psi.block(g).l2_norm();
      norms_test[g] = prods.back().psi.block(g).l2_norm();
    } else {  // L2 norm (of a function)
      norms_trial[g] = std::sqrt(transport.inner_product(
          prods.back().psi.block(g), 
          prods.back().psi.block(g)));
      norms_test[g] = std::sqrt(transport.inner_product(
          prods.back().test.block(g), 
          prods.back().test.block(g)));
    }
  }
  if (!groupwise) {
    norms_trial.assign(norms_trial.size(), 
        std::accumulate(norms_trial.begin(), norms_trial.end(), 0.));
    norms_test.assign(norms_test.size(), 
        std::accumulate(norms_test.begin(), norms_test.end(), 0.));
  }
  // for (int g = (dim % 2); g < mgxs_rom.num_groups; g += 2) {
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    prods.back().psi.block(g) /= norms_trial[g];
    prods.back().phi.block(g) /= norms_trial[g];
    prods.back().streamed.block(g) /= norms_trial[g];
    prods.back().test.block(g) /= norms_test[g];
  }
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::set_products(
    Products &prod) {
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    fixed_src.d2m.vmult(prod.phi.block(g), prod.psi.block(g));
    const auto &tr = dynamic_cast<const TransportBlock<dim, qdim>&>(
        fixed_src.within_groups[g].transport);
    tr.stream(prod.streamed.block(g), prod.psi.block(g));
    transport.vmult_mass_inv(prod.streamed.block(g));
  }
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::set_inner_prods() {
  for (int m = 0; m < prods.size(); ++m)
    set_inner_prod_flux(prods.back().test, prods[m], iprods_flux[m]);
  for (int s = 0; s < srcs.size(); ++s)
    set_inner_prod_src(prods.back().test, srcs[s], iprods_src[s]);
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::set_inner_prod_flux(
      const dealii::BlockVector<double> &test, 
      const Products &prod,
      InnerProducts2D1D &iprod) {
  iprod = 0;
  std::vector<dealii::types::global_dof_index> dof_indices(
      transport.dof_handler.get_fe().dofs_per_cell);
  dealii::BlockVector<double> test_g(transport.quadrature.size(), 
                                     transport.dof_handler.n_dofs());
  dealii::BlockVector<double> mode_g(test_g);
  bool degenerate = transport.quadrature.is_degenerate();
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    double du = 1;  // lethargy width
    iprod.stream_trans[g] += transport.inner_product(
        test.block(g), prod.streamed.block(g)) / du;
    test_g = test.block(g);
    mode_g = prod.psi.block(g);
    for (int n = 0; n < transport.quadrature.size(); ++n) {
      double polar = transport.quadrature.angle(n)[0];
      double wgt = transport.quadrature.weight(n);
      if (!degenerate)
        wgt *= dim == 1 ? std::sqrt(1-std::pow(polar, 2)) : polar;
      iprod.stream_co[g] += (wgt/du) * transport.inner_product(
          test_g.block(n), mode_g.block(n));
    }
    int c = -1;
    for (auto &cell : transport.dof_handler.active_cell_iterators()) {
      if (!cell->is_locally_owned())
        continue;
      c += 1;
      int matl = cell->material_id();
      cell->get_dof_indices(dof_indices);
      for (int n = 0; n < transport.quadrature.size(); ++n) {
        int nn = n * transport.dof_handler.n_dofs();
        for (int i = 0; i < dof_indices.size(); ++i) {
          for (int j = 0; j < dof_indices.size(); ++j) {
            iprod.rxn.total[g][matl] += test.block(g)[nn+dof_indices[i]] *
                                        transport.cell_matrices[c].mass[i][j] *
                                        prod.psi.block(g)[nn+dof_indices[j]] *
                                        transport.quadrature.weight(n);
            for (int gp = 0; gp < mgxs_rom.num_groups; ++gp) {
              iprod.rxn.scatter[g][gp][matl] += 
                  test.block(g)[nn+dof_indices[i]] *
                  transport.cell_matrices[c].mass[i][j] *
                  prod.phi.block(gp)[dof_indices[j]] *
                  transport.quadrature.weight(n);
            }
          }
        }
      }
    }
  }
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::set_inner_prod_src(
    const dealii::BlockVector<double> &test,
    const dealii::BlockVector<double> &src_s,
    std::vector<double> &iprod) {
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    double du = 1;  // lethargy widths
    iprod[g] = transport.inner_product(test.block(g), src_s.block(g)) / du;
  }
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::setup(
    std::vector<InnerProducts2D1D> coeffs_flux,
    const std::vector<std::vector<double>> &coeffs_src,
    const std::vector<std::vector<int>> &materials,
    const Mgxs &mgxs) {
  std::vector<double> denom(coeffs_flux.back().stream_co);
  set_source(coeffs_src);
  set_mgxs(coeffs_flux.back(), materials, mgxs);
  coeffs_flux.pop_back();
  set_residual(coeffs_flux, materials, mgxs);
  rescale_residual(denom);
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::set_source(
    const std::vector<std::vector<double>> &coeffs_src) {
  AssertDimension(coeffs_src.size(), srcs.size());
  src = 0;
  for (int s = 0; s < srcs.size(); ++s)
    for (int g = 0; g < mgxs_rom.num_groups; ++g)
      src.block(g).add(coeffs_src[s][g], srcs[s].block(g));
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::set_residual(
    const std::vector<InnerProducts2D1D> &coeffs_flux,
    const std::vector<std::vector<int>> &materials,
    const Mgxs &mgxs) {
  AssertDimension(coeffs_fluxes.size(), prods.size()-1);
  const int num_groups_nd = mgxs_rom.num_groups;
  const int num_groups_md = coeffs_flux.empty() ?
                            1 : coeffs_flux[0].rxn.num_groups;
  bool mg_nd = num_groups_nd > 1;
  bool mg_md = num_groups_md > 1;
  if (mg_nd)
    AssertDimension(num_groups, num_groups_nd);
  if (mg_md)
    AssertDimension(num_groups, num_groups_md);
  for (int m = 0; m < prods.size()-1; ++m) {
    Mgxs mgxs_m(mgxs_rom);
    mgxs_m *= 0;
    for (int i = 0; i < mgxs_rom.num_materials; ++i) {
      for (int j = 0; j < coeffs_flux[m].rxn.num_materials; ++j) {
        int matl = dim == 1 ? materials[i][j] : materials[j][i];
        for (int g = 0; g < mgxs.num_groups; ++g) {
          int g_nd = mg_nd ? g : 0;
          int g_md = mg_md ? g : 0;
          mgxs_m.total[g_nd][i] += mgxs.total[g][matl] *
                                   coeffs_flux[m].rxn.total[g_md][j];
          for (int gp = 0; gp < mgxs.num_groups; ++gp) {
            int gp_nd = mg_nd ? gp : 0;
            int gp_md = mg_md ? gp : 0;
            mgxs_m.scatter[g_nd][gp_nd][i] += 
                mgxs.scatter[g][gp][matl] *
                coeffs_flux[m].rxn.scatter[g_md][gp_md][j];
          }
        }
      }
    }
    std::vector<double> stream_co(num_groups_nd);
    std::vector<double> stream_trans(num_groups_nd);
    if (num_groups_md == 1) {
      stream_co.assign(num_groups_nd, coeffs_flux[m].stream_co[0]);
      stream_trans.assign(num_groups_nd, coeffs_flux[m].stream_trans[0]);
    } else if (num_groups_md == num_groups_nd) {
      stream_co = coeffs_flux[m].stream_co;
      stream_trans = coeffs_flux[m].stream_trans;
    } else {
      AssertDimension(num_groups_nd, 1);
      stream_co.assign(num_groups_nd, std::accumulate(
          coeffs_flux[m].stream_co.begin(), 
          coeffs_flux[m].stream_co.end(), 
          0.));
      stream_trans.assign(num_groups_nd, std::accumulate(
          coeffs_flux[m].stream_trans.begin(), 
          coeffs_flux[m].stream_trans.end(), 
          0.));
    }
    for (int g = 0; g < mgxs_m.num_groups; ++g) {
      // subtract streamed flux
      for (int n = 0, nk = 0; n < transport.quadrature.size(); ++n)
        for (int k = 0; k < dof_zones.size(); ++k, ++nk)
          src.block(g)[nk] -= stream_co[g] * prods[m].streamed.block(g)[nk];
      // subtract collided flux
      for (int n = 0, nk = 0; n < transport.quadrature.size(); ++n) {
        double leakage_trans = stream_trans[g];
        if (!transport.quadrature.is_degenerate())
          leakage_trans *= dim == 1
              ? std::sqrt(1-std::pow(transport.quadrature.angle(n)[0], 2))
              : transport.quadrature.angle(n)[0];
        for (int k = 0; k < dof_zones.size(); ++k, ++nk)
          src.block(g)[nk] -= (mgxs_m.total[g][dof_zones[k]] + leakage_trans) *
                              prods[m].psi.block(g)[nk];
      }
      for (int gp = 0; gp < mgxs_m.num_groups; ++gp) {
        // add (isotropic) scattered flux
        for (int n = 0, nk = 0; n < transport.quadrature.size(); ++n)
          for (int k = 0; k < dof_zones.size(); ++k, ++nk)
            src.block(g)[nk] += mgxs_m.scatter[g][gp][dof_zones[k]] *
                                prods[m].phi.block(gp)[k];
      }
    }
  }
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::rescale_residual(
    const std::vector<double> &denom) {
  if (denom.size() == 1)
    src /= denom[0];
  else if (src.n_blocks() == 1)
    src /= std::accumulate(denom.begin(), denom.end(), 0.);
  else if (denom.size() == src.n_blocks())
    for (int g = 0; g < denom.size(); ++g)
      src.block(g) /= denom[g];
  else
    AssertThrow(false, dealii::ExcInvalidState());
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::set_mgxs(
    const InnerProducts2D1D &coeff_flux,
    const std::vector<std::vector<int>> &materials,
    const Mgxs &mgxs) {
  const int num_groups_nd = mgxs_rom.num_groups;
  const int num_groups_md = coeff_flux.rxn.num_groups;
  bool mg_nd = num_groups_nd > 1;
  bool mg_md = num_groups_md > 1;
  if (mg_nd)
    AssertDimension(num_groups, num_groups_nd);
  if (mg_md)
    AssertDimension(num_groups, num_groups_md);
  std::vector<double> stream_co(coeff_flux.stream_co);
  if (mg_md && !mg_nd)
    stream_co.assign(num_groups_md, 
        std::accumulate(stream_co.begin(), stream_co.end(), 0.));
  mgxs_rom *= 0;
  for (int i = 0; i < mgxs_rom.num_materials; ++i) {
    for (int j = 0; j < coeff_flux.rxn.num_materials; ++j) {
      int matl = dim == 1 ? materials[i][j] : materials[j][i];
      for (int g = 0; g < mgxs.num_groups; ++g) {
        int g_nd = mg_nd ? g : 0;
        int g_md = mg_md ? g : 0;
        mgxs_rom.total[g_nd][i] += mgxs.total[g][matl] *
                                   coeff_flux.rxn.total[g_md][j] /
                                   stream_co[g_md];
        for (int gp = 0; gp < mgxs.num_groups; ++gp) {
          int gp_nd = mg_nd ? gp : 0;
          int gp_md = mg_md ? gp : 0;
          mgxs_rom.scatter[g_nd][gp_nd][i] += 
              mgxs.scatter[g][gp][matl] *
              coeff_flux.rxn.scatter[g_md][gp_md][j] /
              stream_co[g_md];
        }
      }
    }
  }
  for (int g = 0; g < mgxs_rom.num_groups; ++g)
    const_cast<double&>(fixed_src.within_groups[g].transport.leakage_trans) = 0;
  for (int g = 0; g < mgxs.num_groups; ++g) {
    int g_nd = mg_nd ? g : 0;
    int g_md = mg_md ? g : 0;
    const_cast<double&>(fixed_src.within_groups[g_nd].transport.leakage_trans) += 
        coeff_flux.stream_trans[g_md] / stream_co[g_md];
  }
}

template <int dim, int qdim>
void FixedSource2D1D<dim, qdim>::check_mgxs() {
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    for (int j = 0; j < mgxs_rom.num_materials; ++j) {
      double total = mgxs_rom.total[g][j];
      double scatter = 0;
      for (int gp = 0; gp < mgxs_rom.num_groups; ++gp) {
        scatter += mgxs_rom.scatter[g][gp][j];
      }
      if (transport.quadrature.is_degenerate()) {
        // take credit for transverse leakage

      }
    }
  }
}

template <int dim, int qdim>
double FixedSource2D1D<dim, qdim>::solve() {
  double res_fwd = solve_forward();
  if (is_minimax) {
    solve_adjoint();
  } else {
    prods.back().test = prods.back().psi;
  }
  return res_fwd;
}

template <int dim, int qdim>
double FixedSource2D1D<dim, qdim>::solve_forward() {
  uncollided = 0;
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    fixed_src.within_groups[g].transport.vmult(
        uncollided.block(g), src.block(g), false);
  }
  solver.solve(fixed_src, prods.back().psi, uncollided, fixed_src_gs);
  set_products(prods.back());
  return control.initial_value() / uncollided.l2_norm();
}

template <int dim, int qdim>
double FixedSource2D1D<dim, qdim>::solve_adjoint() {
  uncollided = 0;
  for (int g = 0; g < mgxs_rom.num_groups; ++g) {
    fixed_src.within_groups[g].transport.Tvmult(
        uncollided.block(g), prods.back().psi.block(g), false);
  }
  fixed_src_gs.transposed = !fixed_src_gs.transposed;
  const_cast<bool&>(fixed_src.transposed) = !fixed_src.transposed;
  solver.solve(fixed_src, prods.back().test, uncollided, fixed_src_gs);
  fixed_src_gs.transposed = !fixed_src_gs.transposed;
  const_cast<bool&>(fixed_src.transposed) = !fixed_src.transposed;
  return control.initial_value() / uncollided.l2_norm();
}

template class FixedSource2D1D<1>;
template class FixedSource2D1D<2>;

} // namespace aether::pgd::sn