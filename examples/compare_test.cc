#include "compare_test.h"

template <int dim, int qdim>
void CompareTest<dim, qdim>::WriteUniformFissionSource(
    std::vector<dealii::Vector<double>> &sources_energy,
    std::vector<dealii::BlockVector<double>> &sources_spaceangle) {
  AssertDimension(sources_energy.size(), 0);
  AssertDimension(sources_spaceangle.size(), 0);
  const int num_groups = mgxs->total.size();
  const int num_materials = mgxs->total[0].size();
  sources_energy.resize(num_materials, dealii::Vector<double>(num_groups));
  sources_spaceangle.resize(num_materials, 
      dealii::BlockVector<double>(1, quadrature.size()*dof_handler.n_dofs()));
  // (Energy dependence)
  for (int g = 0; g < num_groups; ++g)
    for (int j = 0; j < num_materials; ++j)
      sources_energy[j][g] = mgxs->chi[g][j];
  // (Spatio-angular dependence)
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); 
      ++cell) {
    cell->get_dof_indices(dof_indices);
    const int j = cell->material_id();
    for (int n = 0; n < quadrature.size(); ++n)
      for (dealii::types::global_dof_index i : dof_indices)
        sources_spaceangle[j][n*dof_handler.n_dofs()+i] = 1;
  }
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::RunPgd(pgd::sn::NonlinearGS &nonlinear_gs, 
                                    const int num_modes,
                                    const int max_iters, 
                                    const double tol, 
                                    const bool do_update,
                                    std::vector<int> &unconverged, 
                                    std::vector<double> &residuals) {
  dealii::BlockVector<double> _;
  for (int m = 0; m < num_modes; ++m) {
    nonlinear_gs.enrich();
    double residual = 0;
    std::cout << "mode " << m << std::endl;
    for (int k = 0; k < max_iters; ++k) {
      bool should_normalize = true;
      try {
        residual = nonlinear_gs.step(_, _, should_normalize, false);
        if (!k)
          residual = std::numeric_limits<double>::infinity();
        std::cout << "picard " << k << " : " << residual << std::endl;
        if (residual < tol)
          break;
      } catch (dealii::SolverControl::NoConvergence &failure) {
        failure.print_info(std::cout);
        break;
      }
    }
    if (residual >= tol) {
      unconverged.push_back(m);
      residuals.push_back(residual);
    }
    nonlinear_gs.finalize();
    if (do_update) {
      // if (m > 0)
      nonlinear_gs.update();
    }
  }
}

template <int dim, int qdim>
double CompareTest<dim, qdim>::ComputeEigenvalue(
    FixedSourceProblem<dim, qdim, pgd::sn::Transport<dim, qdim>> &problem,
    dealii::BlockVector<double> &flux, dealii::BlockVector<double> &source,
    Mgxs &mgxs_problem, double &denominator) {
  denominator = 0;  // power 0
  double numerator = 0;  // power 1
  dealii::Vector<double> scalar(dof_handler.n_dofs());
  dealii::Vector<double> fissioned(dof_handler.n_dofs());
  dealii::Vector<double> dual(dof_handler.n_dofs());
  const int num_groups = source.n_blocks();
  for (int g = 0; g < num_groups; ++g) {
    // from source
    problem.d2m.vmult(scalar, source.block(g));
    problem.transport.collide_ordinate(dual, scalar);
    for (int i = 0; i < dual.size(); ++i)
      denominator += dual[i];
    // from flux
    problem.d2m.vmult(scalar, flux.block(g));
    ScatteringBlock<dim> nu_fission(
        problem.scattering, mgxs_problem.nu_fission[g]);
    nu_fission.vmult(fissioned, scalar);
    problem.transport.collide_ordinate(dual, fissioned);
    for (int i = 0; i < dual.size(); ++i)
      numerator += dual[i];
  }
  return numerator / denominator;
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::GetL2ErrorsDiscrete(
    std::vector<double> &l2_errors,
    const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
    const std::vector<dealii::Vector<double>> &modes_energy,
    const dealii::BlockVector<double> &reference,
    const pgd::sn::Transport<dim, qdim> &transport,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = modes_energy[0].size();
  std::vector<dealii::BlockVector<double>> modal(num_groups,
      dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
  dealii::BlockVector<double> reference_g(modal[0].get_block_indices());
  dealii::Vector<double> diff(dof_handler.n_dofs());
  dealii::Vector<double> diff_l2(diff.size());
  for (int g = 0; g < num_groups; ++g) {
    reference_g = reference.block(g);
    int g_rev = num_groups - 1 - g;
    double width =
        std::log(mgxs->group_structure[g_rev+1]/mgxs->group_structure[g_rev]);
    AssertThrow(width > 0, dealii::ExcInvalidState());
    for (int m = 0; m < l2_errors.size(); ++m) {
      if (m > 0)
        modal[g].add(modes_energy[m-1][g], modes_spaceangle[m-1]);
      for (int n = 0; n < quadrature.size(); ++n) {
        diff = modal[g].block(n);
        diff -= reference_g.block(n);
        transport.collide_ordinate(diff_l2, diff);
        l2_errors[m] += quadrature.weight(n) * (diff * diff_l2) / width;
      }
    }
  }
  for (int m = 0; m < l2_errors.size(); ++m) {
    l2_errors[m] = std::sqrt(l2_errors[m]);
    table.add_value(key, l2_errors[m]);
  }
  table.set_scientific(key, true);
  table.set_precision(key, 16);
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::GetL2ErrorsMoments(
    std::vector<double> &l2_errors,
    const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
    const std::vector<dealii::Vector<double>> &modes_energy,
    const dealii::BlockVector<double> &reference,
    const pgd::sn::Transport<dim, qdim> &transport,
    const DiscreteToMoment<qdim> &d2m,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = modes_energy[0].size();
  dealii::BlockVector<double> mode(quadrature.size(), dof_handler.n_dofs());
  std::vector<dealii::BlockVector<double>> modal(num_groups,
      dealii::BlockVector<double>(1, dof_handler.n_dofs()));
  dealii::BlockVector<double> reference_g_d(mode.get_block_indices());
  dealii::BlockVector<double> reference_g_m(1, dof_handler.n_dofs());
  dealii::Vector<double> diff(dof_handler.n_dofs());
  dealii::Vector<double> diff_l2(diff.size());
  for (int g = 0; g < num_groups; ++g) {
    reference_g_d = reference.block(g);
    d2m.vmult(reference_g_m, reference_g_d);
    int g_rev = num_groups - 1 - g;
    double width =
        std::log(mgxs->group_structure[g_rev+1]/mgxs->group_structure[g_rev]);
    AssertThrow(width > 0, dealii::ExcInvalidState());
    for (int m = 0; m < l2_errors.size(); ++m) {
      if (m > 0) {
        mode.equ(modes_energy[m-1][g], modes_spaceangle[m-1]);
        d2m.vmult_add(modal[g], mode);
      }
      diff = modal[g].block(0);
      diff -= reference_g_m.block(0);
      transport.collide_ordinate(diff_l2, diff);
      l2_errors[m] += (diff * diff_l2) / width;
    }
  }
  for (int m = 0; m < l2_errors.size(); ++m) {
    l2_errors[m] = std::sqrt(l2_errors[m]);
    table.add_value(key, l2_errors[m]);
  }
  table.set_scientific(key, true);
  table.set_precision(key, 16);
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::GetL2Norms(
    std::vector<double> &l2_norms,
    const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
    const std::vector<dealii::Vector<double>> &modes_energy,
    const pgd::sn::Transport<dim, qdim> &transport,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = mgxs->total.size();
  dealii::Vector<double> mode_l2(modes_spaceangle[0].block(0).size());
  for (int m = 0; m < l2_norms.size() - 1; ++m) {
    dealii::Vector<double> summands_energy(modes_energy[m]);
    for (int g = 0; g < summands_energy.size(); ++g) {
      int g_rev = num_groups - 1 - g;
      double width = 
          std::log(mgxs->group_structure[g_rev+1]
                    /mgxs->group_structure[g_rev]);
      AssertThrow(width > 0, dealii::ExcInvalidState());
      summands_energy[g] /= std::sqrt(width);
    }
    double l2_energy = summands_energy.l2_norm();
    double l2_spaceangle = 0;
    for (int n = 0; n < quadrature.size(); ++n) {
      transport.collide_ordinate(mode_l2, modes_spaceangle[m].block(n));
      l2_spaceangle += (modes_spaceangle[m].block(n) * mode_l2)
                        * quadrature.weight(n);
    }
    l2_spaceangle = std::sqrt(l2_spaceangle);
    l2_norms[m] = l2_energy * l2_spaceangle;
    table.add_value(key, l2_norms[m]);
  }
  table.add_value(key, std::nan("a"));
  table.set_scientific(key, true);
  table.set_precision(key, 16);
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::GetL2Residuals(
    std::vector<double> &l2_residuals,
    const std::vector<pgd::sn::Cache> &caches,
    const std::vector<dealii::Vector<double>> &modes_energy,
    dealii::BlockVector<double> residual,
    const pgd::sn::Transport<dim, qdim> &transport,
    const MomentToDiscrete<qdim> &m2d,
    const FixedSourceProblem<dim, qdim> &problem,
    const bool do_stream,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = mgxs->total.size();
  dealii::Vector<double> scattered(quadrature.size()*dof_handler.n_dofs());
  dealii::BlockVector<double> swept(residual.get_block_indices());
  dealii::BlockVector<double> residual_g(
      quadrature.size(), dof_handler.n_dofs());
  dealii::BlockVector<double> swept_g(residual_g.get_block_indices());
  dealii::Vector<double> swept_l2(dof_handler.n_dofs());
  dealii::Vector<double> residual_l2(dof_handler.n_dofs());
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  dealii::Vector<double> streamed_k(dof_indices.size());
  dealii::Vector<double> mass_inv_streamed_k(dof_indices.size());
  std::vector<dealii::BlockVector<double>> boundary_conditions;
  for (int m = 0; m < l2_residuals.size(); ++m) {
    // get norm of residual
    swept = 0;
    for (int g = 0; g < num_groups; ++g) {
      int g_rev = num_groups - 1 - g;
      double width =
          std::log(mgxs->group_structure[g_rev+1]
                    /mgxs->group_structure[g_rev]);
      AssertThrow(width > 0, dealii::ExcInvalidState());
      if (!do_stream) {
        residual_g = residual.block(g);
      } else {
        std::vector<double> cross_sections(mgxs->total[g].size());
        transport.vmult(swept.block(g), residual.block(g), 
                        cross_sections, boundary_conditions);
        swept_g = swept.block(g);
      }
      for (int n = 0; n < quadrature.size(); ++n) {
        if (!do_stream) {
          transport.collide_ordinate(residual_l2, residual_g.block(n));
          l2_residuals[m] += (residual_g.block(n) * residual_l2) 
                            * quadrature.weight(n) / width;
        } else {
          transport.collide_ordinate(swept_l2, swept_g.block(n));
          l2_residuals[m] += (swept_g.block(n) * swept_l2) 
                            * quadrature.weight(n) / width;
        }
      }
    }
    l2_residuals[m] = std::sqrt(l2_residuals[m]);
    table.add_value(key, l2_residuals[m]);
    if (m == l2_residuals.size()-1)
      continue;
    // update residual
    m2d.vmult(scattered, caches[m].moments.block(0));
    for (int g = 0; g < num_groups; ++g) {
      int c = 0;
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
            ++cell, ++c) {
        if (!cell->is_locally_owned()) {
          --c;
          continue;
        }
        cell->get_dof_indices(dof_indices);
        const int j = cell->material_id();
        dealii::FullMatrix<double> mass = transport.cell_matrices[c].mass;
        mass.gauss_jordan();
        for (int n = 0; n < quadrature.size(); ++n) {
          for (int i = 0; i < dof_indices.size(); ++i) {
            const dealii::types::global_dof_index ni =
                n * dof_handler.n_dofs() + dof_indices[i];
            streamed_k[i] = caches[m].streamed.block(0)[ni];
          }
          mass.vmult(mass_inv_streamed_k, streamed_k);
          for (int i = 0; i < dof_indices.size(); ++i) {
            const dealii::types::global_dof_index ni =
                n * dof_handler.n_dofs() + dof_indices[i];
            double dof_m = 
                mass_inv_streamed_k[i] * modes_energy[m][g];
            dof_m += mgxs->total[g][j] * caches[m].mode.block(0)[ni] 
                      * modes_energy[m][g];
            for (int gp = 0; gp < num_groups; ++gp)
              dof_m += mgxs->scatter[g][gp][j] * scattered[ni]
                        * modes_energy[m][gp];
            residual.block(g)[ni] -= dof_m;
          }
        }
      }
    }
  }
  table.set_scientific(key, true);
  table.set_precision(key, 16);
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::GetL2ResidualsFull(
    std::vector<double> &l2_residuals,
    const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
    const std::vector<dealii::Vector<double>> &modes_energy,
    dealii::BlockVector<double> &uncollided,
    const pgd::sn::Transport<dim, qdim> &transport,
    const FixedSourceProblem<dim, qdim> &problem,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = mgxs->total.size();
  dealii::BlockVector<double> flux(uncollided.get_block_indices());
  dealii::BlockVector<double> residual(uncollided);
  dealii::BlockVector<double> residual_g(
      quadrature.size(), dof_handler.n_dofs());
  dealii::Vector<double> residual_l2(dof_handler.n_dofs());
  dealii::Vector<double> mode_spaceangle(
      quadrature.size() * dof_handler.n_dofs());
  for (int m = 0; m < l2_residuals.size(); ++m) {
    for (int g = 0; g < num_groups; ++g) {
      int g_rev = num_groups - 1 - g;
      double width =
          std::log(mgxs->group_structure[g_rev+1]
                    /mgxs->group_structure[g_rev]);
      AssertThrow(width > 0, dealii::ExcInvalidState());
      residual_g = residual.block(g);
      for (int n = 0; n < quadrature.size(); ++n) {
        transport.collide_ordinate(residual_l2, residual_g.block(n));
        l2_residuals[m] += (residual_g.block(n) * residual_l2)
                            * quadrature.weight(n) / width;
      }
    }
    l2_residuals[m] = std::sqrt(l2_residuals[m]);
    table.add_value(key, l2_residuals[m]);
    if (m == l2_residuals.size()-1)
      continue;
    for (int g = 0; g < num_groups; ++g) {
      mode_spaceangle = modes_spaceangle[m];
      flux.block(g).add(modes_energy[m][g], mode_spaceangle);
    }
    residual = flux;
    problem.fixed_source.vmult(residual, flux);
    residual.sadd(-1, 1, uncollided);
  }
  table.set_scientific(key, true);
  table.set_precision(key, 16);
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::ComputeSvd(
    std::vector<dealii::BlockVector<double>> &svecs_spaceangle,
    std::vector<dealii::Vector<double>> &svecs_energy,
    const dealii::BlockVector<double> &flux,
    const Transport<dim, qdim> &transport) {
  AssertDimension(svecs_spaceangle.size(), 0);
  AssertDimension(svecs_energy.size(), 0);
  const int num_groups = flux.n_blocks();
  const int num_qdofs = flux.block(0).size();
  std::cout << "initialize flux matrix\n";
  dealii::LAPACKFullMatrix_<double> flux_matrix(num_qdofs, num_groups);
  std::cout << "initialized flux matrix\n";
  std::vector<dealii::FullMatrix<double>> masses_cho(
      dof_handler.get_triangulation().n_active_cells());
  for (int c = 0; c < masses_cho.size(); ++c)
    masses_cho[c].cholesky(transport.cell_matrices[c].mass);
  double lowest = 1e-5;
  for (int g = 0; g < num_groups; ++g) {
    int g_rev = num_groups - 1 - g;
    double lower = mgxs->group_structure[g_rev] > 0 ?
                    mgxs->group_structure[g_rev] : lowest;
    double width =
        std::log(mgxs->group_structure[g_rev+1]/lower);
    for (int n = 0; n < quadrature.size(); ++n) {
      for (int c = 0; c < masses_cho.size(); ++c) {
        int nc = n * dof_handler.n_dofs() 
                  + c * dof_handler.get_fe().n_dofs_per_cell();
        dealii::FullMatrix<double> &mass_cho = masses_cho[c];
        for (int i = 0; i < mass_cho.m(); ++i) {
          for (int j = 0; j < mass_cho.n(); ++j) {
            flux_matrix(nc+i, g) += mass_cho[i][j] * flux.block(g)[nc+j]
                                    * std::sqrt(quadrature.weight(n))
                                    / std::sqrt(width);
          }
        }
      }
    }
  }
  // invert masses cholesky
  for (auto &mass_cho : masses_cho)
    mass_cho.gauss_jordan();
  // compute svd and post-process
  std::cout << "compute svd\n";
  flux_matrix.compute_svd('S');
  std::cout << "computed svd\n";
  const int num_svecs = std::min(num_groups, num_qdofs);
  AssertDimension(num_qdofs, quadrature.size() * dof_handler.n_dofs());
  svecs_spaceangle.resize(num_svecs, 
      dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
  svecs_energy.resize(num_svecs, dealii::Vector<double>(num_groups));
  for (int s = 0; s < num_svecs; ++s) {
    for (int n = 0; n < quadrature.size(); ++n) {
      for (int c = 0; c < masses_cho.size(); ++c) {
        int nc = n * dof_handler.n_dofs() 
                  + c * dof_handler.get_fe().n_dofs_per_cell();
        dealii::FullMatrix<double> mass_cho_inv = masses_cho[c];
        for (int i = 0; i < mass_cho_inv.m(); ++i) {
          for (int j = 0; j < mass_cho_inv.n(); ++j) {
            svecs_spaceangle[s][nc+i] += mass_cho_inv[i][j] 
                                          * flux_matrix.get_svd_u()(nc+j, s)
                                          / std::sqrt(quadrature.weight(n));
          }
        }
      }
    }
    for (int g = 0; g < num_groups; ++g) {
      int g_rev = num_groups - 1 - g;
      double lower = mgxs->group_structure[g_rev] > 0 ? 
                      mgxs->group_structure[g_rev] : lowest;
      double width =
          std::log(mgxs->group_structure[g_rev+1]/lower);
      svecs_energy[s][g] = flux_matrix.get_svd_vt()(s, g) * std::sqrt(width);
    }
    svecs_energy[s] *= flux_matrix.singular_value(s);
  }
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::Compare(int num_modes,
                                     const int max_iters_nonlinear,
                                     const double tol_nonlinear,
                                     const int max_iters_fullorder,
                                     const double tol_fullorder,
                                     const bool do_update,
                                     const bool precomputed_full,
                                     const bool precomputed_pgd,
                                     const bool do_eigenvalue,
                                     const bool full_only,
                                     int num_modes_s,
                                     const bool guess_svd,
                                     const bool guess_spatioangular) {
  const int num_groups = mgxs->total.size();
  const int num_materials = mgxs->total[0].size();
  // Create sources
  std::vector<dealii::Vector<double>> sources_energy;
  std::vector<dealii::BlockVector<double>> sources_spaceangle;
  WriteUniformFissionSource(sources_energy, sources_spaceangle);
  const int num_sources = sources_energy.size();
  // Create boundary conditions
  std::vector<std::vector<dealii::BlockVector<double>>> 
      boundary_conditions_one(1);
  std::vector<std::vector<dealii::BlockVector<double>>> 
      boundary_conditions(num_groups);
  // Run full order
  dealii::BlockVector<double> flux_full(
      num_groups, quadrature.size()*dof_handler.n_dofs());
  dealii::BlockVector<double> source_full(flux_full.get_block_indices());
  for (int g = 0; g < num_groups; ++g)
    for (int j = 0; j < num_sources; ++j)
      source_full.block(g).add(
          sources_energy[j][g], sources_spaceangle[j].block(0));
  FissionProblem<dim, qdim> problem_full(
      dof_handler, quadrature, *mgxs, boundary_conditions);
  double eigenvalue_full = 0;
  std::string filebase = "";
  using StrStrInt = std::tuple<std::string, std::string, int>;
  auto this_with_params = dynamic_cast<const ::testing::WithParamInterface<
      StrStrInt>*>(this);
  if (this_with_params != nullptr) {
    StrStrInt param = this_with_params->GetParam();
    filebase = std::get<0>(param) + "_" + std::get<1>(param);
  }
  std::string filename_full = filebase + "_full.h5";
  if (do_eigenvalue)
    filename_full = filebase + "_k_full.h5";
  namespace HDF5 = dealii::HDF5;
  if (precomputed_full) {
    HDF5::File file(filename_full, HDF5::File::FileAccessMode::open);
    flux_full = file.open_dataset("flux_full").read<dealii::Vector<double>>();
    if (do_eigenvalue) {
      eigenvalue_full = file.get_attribute<double>("k_eigenvalue");
      std::cout << "eigenvalue_full: " << eigenvalue_full << "\n";
    }
  } else {
    std::vector<double> history_data;
    // run problem
    if (do_eigenvalue) {
      eigenvalue_full = RunFullOrderCriticality(
          flux_full, source_full, problem_full, 
          max_iters_fullorder, tol_fullorder, &history_data);
    } else {
      RunFullOrder(flux_full, source_full, problem_full, 
                  max_iters_fullorder, tol_fullorder, &history_data);
    }
    this->WriteFlux(flux_full, history_data, filename_full, eigenvalue_full);
  }
  if (full_only)
    return;
  double sum = 0;
  for (int i = 0; i < flux_full.size(); ++i)
    sum += flux_full[i];
  if (sum < 0)
    flux_full *= -1;
  this->PlotFlux(flux_full, problem_full.d2m, mgxs->group_structure, "full");
  // Compute svd of full order
  std::cout << "compute svd\n";
  std::vector<dealii::BlockVector<double>> svecs_spaceangle;
  std::vector<dealii::Vector<double>> svecs_energy;
  ComputeSvd(svecs_spaceangle, svecs_energy, flux_full, 
              problem_full.transport);
  int num_svecs = svecs_spaceangle.size();
  if (num_svecs > num_modes) {
    svecs_spaceangle.resize(num_modes);
    svecs_energy.resize(num_modes);
    num_svecs = num_modes;
  }
  // Run pgd model
  Mgxs mgxs_one(1, num_materials, 1);
  Mgxs mgxs_pseudo(1, num_materials, 1);
  for (int j = 0; j < num_materials; ++j) {
    bool is_fissionable = true;
    // for (int g = 0; g < num_groups; ++g) {
    //   if (mgxs->chi[g][j] != 0) {
    //     is_fissionable = true;
    //     break;
    //   }
    // }
    mgxs_one.total[0][j] = 1;
    mgxs_one.scatter[0][0][j] = 1;
    mgxs_one.chi[0][j] = is_fissionable ? 1 : 0;
    mgxs_one.nu_fission[0][j] = is_fissionable ? 1 : 0;
    mgxs_pseudo.chi[0][j] = 1;
  }
  using TransportType = pgd::sn::Transport<dim, qdim>;
  using TransportBlockType = pgd::sn::TransportBlock<dim, qdim>;
  FissionProblem<dim, qdim, TransportType, TransportBlockType> problem(
      dof_handler, quadrature, mgxs_pseudo, boundary_conditions_one);
  pgd::sn::FixedSourceP fixed_source_p(
      problem.fixed_source, mgxs_pseudo, mgxs_one, sources_spaceangle);
  pgd::sn::EnergyMgFull energy_mg(*mgxs, sources_energy);
  pgd::sn::FissionSourceP fission_source_p(
    problem.fixed_source, problem.fission, mgxs_pseudo, mgxs_one);
  pgd::sn::EnergyMgFiss energy_fiss(*mgxs);
  const std::string filename_pgd = filebase + "_pgd_" +
      (do_update ? "update" : "prog") + ".h5";
  if (precomputed_pgd) {
    // read from file
    HDF5::File file(filename_pgd, HDF5::File::FileAccessMode::open);
    for (int m = 0; m < num_modes; ++m) {
      const std::string mm = std::to_string(m);
      fixed_source_p.enrich(0);
      fixed_source_p.caches.back().mode = file.open_dataset(
          "modes_spaceangle"+mm).read<dealii::Vector<double>>();
      fixed_source_p.set_cache(fixed_source_p.caches.back());
      energy_mg.modes.push_back(file.open_dataset(
          "modes_energy"+mm).read<dealii::Vector<double>>());
    }
  } else {
    // run pgd
    std::vector<pgd::sn::LinearInterface*> linear_ops;
    std::unique_ptr<pgd::sn::NonlinearGS> nonlinear_gs;
    if (do_eigenvalue) {
      linear_ops = {&energy_fiss, &fission_source_p};
      nonlinear_gs =
          std::make_unique<pgd::sn::EigenGS>(linear_ops, num_materials, 1);
      auto eigen_gs = dynamic_cast<pgd::sn::EigenGS*>(nonlinear_gs.get());
      eigen_gs->initialize_guess();
    } else {
      linear_ops = {&energy_mg, &fixed_source_p};
      nonlinear_gs = std::make_unique<pgd::sn::NonlinearGS>(
          linear_ops, num_materials, 1, num_sources);
    }
    std::vector<int> unconverged;
    std::vector<double> residuals;
    std::cout << "run pgd\n";
    RunPgd(*nonlinear_gs, num_modes, max_iters_nonlinear, tol_nonlinear,
            do_update, unconverged, residuals);
    std::cout << "done running pgd\n";
    std::ofstream unconverged_txt;
    unconverged_txt.open(this->GetTestName()+"_unconverged.txt", 
                        std::ios::trunc);
    for (int u = 0; u < unconverged.size(); ++u)
      unconverged_txt << unconverged[u] << " " << residuals[u] << std::endl;
    unconverged_txt.close();
    std::cout << "wrote unconverged\n";
    // write to file
    HDF5::File file(filename_pgd, HDF5::File::FileAccessMode::create);
    for (int m = 0; m < num_modes; ++m) {
      const std::string mm = std::to_string(m);
      file.write_dataset(
          "modes_spaceangle"+mm, fixed_source_p.caches[m].mode.block(0));
      file.write_dataset("modes_energy"+mm, energy_mg.modes[m]);
    }
  }
  std::vector<dealii::BlockVector<double>> modes_spaceangle(num_modes,
      dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
  for (int m = 0; m < num_modes; ++m) {
    modes_spaceangle[m] = fixed_source_p.caches[m].mode.block(0);
    if (guess_svd) {
      fixed_source_p.caches[m].mode.block(0) = svecs_spaceangle[m]; 
      energy_mg.modes[m] = svecs_energy[m];
    }
  }
  if (num_modes_s > 0) {
    for (int m = num_modes_s; m < num_modes; ++m)
      fixed_source_p.caches.pop_back();
    energy_mg.modes.resize(num_modes_s);
    std::vector<std::vector<pgd::sn::InnerProducts>> inner_products_x( 
        num_modes_s, std::vector<pgd::sn::InnerProducts>(
          num_modes_s, pgd::sn::InnerProducts(num_materials, 1)));
    std::vector<std::vector<double>> inner_products_b(num_modes_s,
        std::vector<double>(num_sources));
    if (false && num_modes_s != num_modes) {
      for (int m = 0; m < num_modes_s; ++m) {
        fixed_source_p.get_inner_products(inner_products_x[m], 
                                          inner_products_b[m], m, 0);
      }
      energy_mg.update(inner_products_x, inner_products_b);
    }
    pgd::sn::FissionSProblem<dim, qdim> subspace_problem(
        dof_handler, quadrature, mgxs_one, boundary_conditions, num_modes_s);
    // normalize
    for (int m = 0; m < num_modes_s; ++m) {
      double norm_m = subspace_problem.transport.inner_product(
          fixed_source_p.caches[m].mode.block(0), 
          fixed_source_p.caches[m].mode.block(0));
      fixed_source_p.caches[m].mode /= norm_m;
      energy_mg.modes[m] *= norm_m;
    }
    for (int m = 0; m < num_modes_s; ++m) {
      energy_mg.get_inner_products(
          inner_products_x[m], inner_products_b[m], m, 0);
    }
    dealii::BlockVector<double> modes(
        num_modes_s, quadrature.size()*dof_handler.n_dofs());
    for (int m = 0; m < num_modes_s; ++m) {
      if (guess_spatioangular)
        modes.block(m) = m == 0 ? 1 : 0;
      else
        modes.block(m) = fixed_source_p.caches[m].mode.block(0);
    }
    double rayleigh = 0;
    if (guess_spatioangular) {
      double norm_0 = subspace_problem.transport.inner_product(
          modes.block(0), modes.block(0));
      modes.block(0) /= norm_0;
      dealii::BlockVector<double> ax(modes);
      dealii::BlockVector<double> bx(modes);
      subspace_problem.set_cross_sections(inner_products_x);
      subspace_problem.fission_s.vmult(ax, modes);
      subspace_problem.fixed_source_s.vmult(bx, modes);
      rayleigh = (modes * ax) / (modes * bx);
      std::cout << "rayleigh " << rayleigh << "?\n";
      dealii::IterationNumberControl control_sa(50, 0);
      dealii::SolverFGMRES<dealii::BlockVector<double>> solver_sa(control_sa);
      solver_sa.solve(subspace_problem.fixed_source_s, modes, ax, 
                      subspace_problem.fixed_source_s_gs);
      for (int m = 0; m < num_modes_s; ++m) {
        double norm_m = subspace_problem.transport.inner_product(
            modes.block(m), modes.block(m));
        modes.block(m) /= norm_m;
      }
      subspace_problem.fission_s.vmult(ax, modes);
      subspace_problem.fixed_source_s.vmult(bx, modes);
      double rayleigh2 = (modes * ax) / (modes * bx);
      std::cout << "rayleigh2 " << rayleigh2 << "?\n";
    }
    double norm = 0;
    for (int m = 0; m < num_modes_s; ++m) {
      for (int g = 0; g < num_groups; ++g) {
        norm += std::pow(energy_mg.modes[m][g], 2);
      }
    }
    std::cout << "norm: " << norm << ", sqrt: " << std::sqrt(norm) << std::endl;
    norm = std::sqrt(norm);
    for (int m = 0; m < num_modes_s; ++m) {
      energy_mg.modes[m] /= norm;
    }
    modes /= modes.l2_norm();
    for (int m = 0; m < num_modes_s; ++m) {
      energy_fiss.modes.push_back(energy_mg.modes[m]);
    }
    std::vector<std::vector<std::vector<aether::pgd::sn::InnerProducts>>>
        coefficients;
    // do JFNK
    std::vector<aether::pgd::sn::SubspaceEigen*> eigen_ops = 
        {&subspace_problem, &energy_fiss};
    aether::pgd::sn::SubspaceJacobianFD jacobian(
        eigen_ops, num_modes_s, num_materials, 1);
    std::cout << "init'd\n";
    dealii::BlockVector<double> modes_all(std::vector<unsigned int>(
        {modes.size(), energy_fiss.modes.size()*energy_fiss.modes[0].size(), 
          1}));
    for (int i = 0; i < modes.size(); ++i)
      modes_all.block(0)[i] = modes[i];
    for (int m = 0; m < num_modes_s; ++m)
      for (int g = 0; g < num_groups; ++g)
        modes_all.block(1)[m*num_groups+g] = energy_fiss.modes[m][g];
    modes_all.block(modes_all.n_blocks()-1)[0] = rayleigh;  // initial k
    dealii::BlockVector<double> step(modes_all);
    aether::pgd::sn::SubspaceJacobianPC<dim, qdim> jacobian_pc(
        subspace_problem, energy_fiss, jacobian.inner_products_unperturbed,
        jacobian.k_eigenvalue);
    const int jfnk_iters = 10;
    for (int i = 0; i <= jfnk_iters; ++i) {
      std::cout << "setting modes " << modes_all.l2_norm() << "\n";
      // modes_all.block(0) /= modes_all.block(0).l2_norm();
      jacobian.set_modes(modes_all);
      jacobian_pc.modes = modes_all;
      for (int m = 0; m < num_modes_s; ++m)
        for (int g = 0; g < num_groups; ++g)
          energy_fiss.modes[m][g] = modes_all.block(1)[m*num_groups+g];
      double tol = 1e-12;
      double k_energy = 0;
      k_energy = 
          energy_fiss.update(jacobian.inner_products_unperturbed[0], tol);
      std::cout << "k-energy: " << k_energy << "\n";
      // step = 0;
      for (int m = 0; m < num_modes_s; ++m)
        for (int g = 0; g < num_groups; ++g)
          modes_all.block(1)[m*num_groups+g] = energy_fiss.modes[m][g];
          // step = energy_fiss.modes[m][g] - modes_all.block(1)[m*num_groups+g];
      // modes_all.block(1) /= modes_all.block(1).l2_norm();
      if (k_energy != 0) {
        modes_all.block(modes_all.n_blocks()-1)[0] = k_energy;
        // step.block(step.n_blocks()-1)[0] = 
            // k_energy - modes_all.block(modes_all.n_blocks()-1)[0];
      }
      jacobian.set_modes(modes_all);
      jacobian_pc.modes = modes_all;
      std::cout << "set modes\n";
      if (i == jfnk_iters)
        continue;
      if (false) {  // do this bad thing
        modes = modes_all.block(0);
        dealii::BlockVector<double> modes_copy(modes);
        modes_copy = modes;
        dealii::BlockVector<double> ax(modes);
        dealii::BlockVector<double> bx(modes);
        subspace_problem.set_cross_sections(
          jacobian.inner_products_unperturbed[1]);
        for (int foo = 0; foo < 1; ++foo) {
          subspace_problem.fission_s.vmult(ax, modes);
          subspace_problem.fixed_source_s.vmult(bx, modes);
          dealii::IterationNumberControl control_sa(10, 0);
          control_sa.enable_history_data();
          dealii::SolverFGMRES<dealii::BlockVector<double>> solver_sa(control_sa);
          solver_sa.solve(subspace_problem.fixed_source_s, modes, ax, 
                          subspace_problem.fixed_source_s_gs);
          // aether::pgd::sn::ShiftedS<2> shifted(subspace_problem.fission_s,
          //                                      subspace_problem.fixed_source_s);
          // shifted.shift = k_energy;
          // subspace_problem.fission_s_gs.set_shift(k_energy);
          // solver_sa.solve(shifted, modes, bx, subspace_problem.fission_s_gs);
          std::cout << "B^-1: ";
          for (double res : control_sa.get_history_data())
            std::cout << res << ", ";
          std::cout << "\n";
          modes.add(-k_energy, modes_copy);
        }
        if (false) {
          dealii::IterationNumberControl control_sa(10, 0);
          control_sa.enable_history_data();
          dealii::SolverFGMRES<dealii::BlockVector<double>> solver_sa(control_sa);
          aether::pgd::sn::FissionSourceShiftedS<2> shifted(
              subspace_problem.fission_s, subspace_problem.fixed_source_s,
              subspace_problem.fission_s_gs);
          shifted.shift = k_energy;
          solver_sa.solve(shifted, modes, modes_copy, dealii::PreconditionIdentity());
          // shifted.vmult(modes, modes_copy);
          std::cout << "sinvert: ";
          for (double res : control_sa.get_history_data())
            std::cout << res << ", ";
          std::cout << "\n";
        }
        modes /= modes.l2_norm();
        modes_all.block(0) = modes;
        jacobian.set_modes(modes_all);
        jacobian_pc.modes = modes_all;
        continue;
      }
      // set initial guess
      // step = modes_all;
      // step *= -1;
      // step[step.size()-1] *= 1e-5;
      // step /= step.l2_norm();
      // step *= jacobian.residual_unperturbed.l2_norm();
      step = jacobian.residual_unperturbed;
      // solve
      dealii::IterationNumberControl control(5, 0);
      control.enable_history_data();
      dealii::SolverFGMRES<dealii::BlockVector<double>> solver(control);
      solver.solve(jacobian, step, jacobian.residual_unperturbed, 
                    jacobian_pc 
                    // dealii::PreconditionIdentity()
                    );
      if (false) { // do l2 line search
        double norm_0 = jacobian.residual_unperturbed.l2_norm();
        modes_all.add(0.5, step);
        jacobian.set_modes(modes_all);
        double norm_half = jacobian.residual_unperturbed.l2_norm();
        modes_all.add(0.5, step);
        jacobian.set_modes(modes_all);
        double norm_1 = jacobian.residual_unperturbed.l2_norm();
        norm_0 *= norm_0;
        norm_half *= norm_half;
        norm_1 *= norm_1;
        double grad_1 = 3*norm_1 - 4*norm_half + norm_0;
        double grad_0 = norm_1 - 4*norm_half + 3*norm_0;
        grad_1 /= 2;
        grad_0 /= 2;
        double step_size = 1 - (grad_1*0.5) / (grad_1-grad_0);
        std::cout << "STEP SIZE: " << step_size << "\n";
        modes_all.add(step_size-1, step);
      } else {
        modes_all.add(1, step);
      }
      for (double res : control.get_history_data())
        std::cout << res << ", ";
      std::cout << "\n";
    }
    jacobian.set_modes(modes_all);
    jacobian_pc.modes = modes_all;
    const int num_qdofs = modes_spaceangle[0].size();
    for (int m = 0; m < num_modes_s; ++m) {
      // energy_mg.modes[m] = energy_fiss.modes[m];
      for (int g = 0; g < num_groups; ++g)
        energy_mg.modes[m][g] = modes_all.block(1)[m*num_groups+g];
      for (int i = 0; i < num_qdofs; ++i) {
        modes_spaceangle[m][i] = modes_all.block(0)[m*num_qdofs+i];
        modes.block(m)[i] = modes_all.block(0)[m*num_qdofs+i];
      }
    }
    num_modes = num_modes_s;
    dealii::BlockVector<double> flux_pgd(flux_full);
    flux_pgd = 0;
    for (int m = 0; m < num_modes; ++m) {
      for (int g = 0; g < num_groups; ++g) {
        flux_pgd.block(g).add(energy_mg.modes[m][g], modes.block(m));
      }
    }
    double norm_expanded = flux_pgd.l2_norm();
    for (int m = 0; m < num_modes; ++m)
      energy_mg.modes[m] /= norm_expanded;
    flux_pgd /= norm_expanded;
    sum = 0;
    for (int i = 0; i < flux_pgd.size(); ++i)
      sum += flux_pgd[i];
    if (sum < 0) {
      flux_pgd *= -1;
      for (int m = 0; m < num_modes; ++m)
        energy_mg.modes[m] *= -1;
    }
    double k_eigenvalue_pgd_s = modes_all.block(modes_all.n_blocks()-1)[0];
    // store subspace pgd
    const std::string filename_pgd_s = 
      this->GetTestName() + "_pgd_s_M" + std::to_string(num_modes_s) + ".h5";
    HDF5::File file(filename_pgd_s, HDF5::File::FileAccessMode::create);
    file.set_attribute("k_eigenvalue", k_eigenvalue_pgd_s);
    for (int m = 0; m < num_modes; ++m) {
      const std::string mm = std::to_string(m);
      file.write_dataset("modes_spaceangle"+mm, modes.block(m));
      file.write_dataset("modes_energy"+mm, energy_mg.modes[m]);
    }
    // stored subspace pgd
    problem_full.fission.vmult(source_full, flux_pgd, false);
    source_full /= k_eigenvalue_pgd_s;
    dealii::BlockVector<double> ax_full(flux_pgd);
    dealii::BlockVector<double> bx_full(flux_pgd);
    problem_full.fixed_source.vmult(bx_full, flux_pgd);
    problem_full.fission.vmult(ax_full, flux_pgd);
    double rayleigh_full = (flux_pgd * ax_full) / (flux_pgd * bx_full);
    std::cout << "rayleigh_full: " << rayleigh_full << "\n";
  }
  dealii::ConvergenceTable table;
  std::vector<double> l2_errors_svd_d(num_svecs+1);
  std::vector<double> l2_errors_svd_m(num_svecs+1);
  std::vector<double> l2_errors_d(num_modes+1);
  std::vector<double> l2_errors_m(num_modes+1);
  std::vector<double> l2_residuals(num_modes+1);
  std::vector<double> l2_residuals_streamed(num_modes+1);
  std::vector<double> l2_residuals_swept(num_modes+1);
  std::vector<double> l2_norms(num_modes+1);
  std::cout << "get l2 errors svd\n";
  GetL2ErrorsDiscrete(l2_errors_svd_d, svecs_spaceangle, svecs_energy, 
                      flux_full, problem.transport, table, "error_svd_d");
  GetL2ErrorsMoments(l2_errors_svd_m, svecs_spaceangle, svecs_energy, 
                      flux_full, problem.transport, problem.d2m, table, 
                      "error_svd_m");
  for (int pad = 0; pad < num_modes - num_svecs; ++pad) {
    table.add_value("error_svd_d", std::nan("p"));
    table.add_value("error_svd_m", std::nan("p"));
  }
  std::cout << "get l2 errors pgd\n";
  GetL2ErrorsDiscrete(l2_errors_d, modes_spaceangle, energy_mg.modes, 
                      flux_full, problem.transport, table, "error_d");
  GetL2ErrorsMoments(l2_errors_m, modes_spaceangle, energy_mg.modes, 
                      flux_full, problem.transport, problem.d2m, table, 
                      "error_m");
  GetL2Norms(l2_norms, modes_spaceangle, energy_mg.modes, problem.transport,
              table, "norm");
  if (num_groups < 800) {
  std::cout << "residuals\n";
  GetL2Residuals(l2_residuals, fixed_source_p.caches, energy_mg.modes, 
                  source_full, problem.transport, problem.m2d, problem_full,
                  false, table, "residual");
  std::cout << "residuals streamed\n";
  GetL2Residuals(l2_residuals_streamed, fixed_source_p.caches, 
                  energy_mg.modes, source_full, problem.transport, problem.m2d,
                  problem_full, true, table, "residual_streamed");
  dealii::BlockVector<double> uncollided(source_full.get_block_indices());
  problem_full.sweep_source(uncollided, source_full);
  GetL2ResidualsFull(l2_residuals_swept, modes_spaceangle, energy_mg.modes, 
                      uncollided, problem.transport, problem_full, table, 
                      "residual_swept");
  }
  if (num_modes_s > 0)
    this->WriteConvergenceTable(table, "_M"+std::to_string(num_modes_s));
  else
    this->WriteConvergenceTable(table);
}

template class CompareTest<2>;
