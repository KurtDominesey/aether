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
                                    std::vector<double> &residuals,
                                    std::vector<double> *eigenvalues) {
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
      double eigenvalue = nonlinear_gs.update();
      if (eigenvalues != nullptr)
        eigenvalues->push_back(eigenvalue);
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
void CompareTest<dim, qdim>::GetL2ErrorsFissionSource(
    std::vector<double> &l2_errors,
    const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
    const std::vector<dealii::Vector<double>> &modes_energy,
    const dealii::BlockVector<double> &reference,
    const pgd::sn::Transport<dim, qdim> &transport,
    const DiscreteToMoment<qdim> &d2m,
    const Production<dim> &production,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = modes_energy[0].size();
  dealii::BlockVector<double> reference_g_d(
      quadrature.size(), dof_handler.n_dofs());
  dealii::BlockVector<double> reference_g_m(1, dof_handler.n_dofs());
  dealii::BlockVector<double> reference_m(num_groups, dof_handler.n_dofs());
  dealii::Vector<double> reference_q(dof_handler.n_dofs());
  for (int g = 0; g < num_groups; ++g) {
    reference_g_d = reference.block(g);
    d2m.vmult(reference_g_m, reference_g_d);
    reference_m.block(g) = reference_g_m;
  }
  production.vmult(reference_q, reference_m);
  dealii::BlockVector<double> mode_g_d(quadrature.size(), dof_handler.n_dofs());
  dealii::BlockVector<double> mode_g_m(1, dof_handler.n_dofs());
  dealii::BlockVector<double> mode_m(num_groups, dof_handler.n_dofs());
  dealii::Vector<double> modal_q(dof_handler.n_dofs());
  dealii::Vector<double> diff(dof_handler.n_dofs());
  dealii::Vector<double> diff_l2(diff.size());
  double l1_norm = 0;
  std::vector<int> indices;
  std::vector<dealii::Vector<double>> diffs;
  for (int m = 0; m < l2_errors.size(); ++m) {
    for (int g = 0; g < num_groups; ++g) {
      if (m > 0) {
        mode_g_d.equ(modes_energy[m-1][g], modes_spaceangle[m-1]);
        d2m.vmult(mode_g_m, mode_g_d);
        mode_m.block(g) = mode_g_m;
      }
    }
    production.vmult_add(modal_q, mode_m);
    diff = modal_q;
    diff -= reference_q;
    if ((m % 10) == 0 || m == 1) {  // save diff to plot
      indices.push_back(m);
      diffs.push_back(diff);
    }
    transport.collide_ordinate(diff_l2, diff);
    l2_errors[m] = std::sqrt(diff * diff_l2);
    if (m == 0)
      for (int i = 0; i < diff_l2.size(); ++i)
        l1_norm -= diff_l2[i];
    table.add_value(key, l2_errors[m]);
  }
  table.set_scientific(key, true);
  table.set_precision(key, 16);
  // plot diffs
  diff = 0;
  for (int i = 0; i < reference_q.size(); ++i) {
    if (reference_q[i] > 0) {
      diff[i] = 1;
      continue;
    }
    reference_q[i] = std::nan("z");
    for (auto &diff_j: diffs)
      diff_j[i] = std::nan("z");
  }
  transport.collide_ordinate(diff_l2, diff);
  double area = 0;
  for (int i = 0; i < diff_l2.size(); ++i)
    area += diff_l2[i];
  l1_norm /= area;
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(this->dof_handler);
  reference_q /= l1_norm;
  data_out.add_data_vector(reference_q, "ref");
  for (int i = 0; i < indices.size(); ++i) {
    int m = indices[i];
    diffs[i] /= l1_norm;
    data_out.add_data_vector(diffs[i], std::to_string(m));
  }
  data_out.build_patches();
  std::string name = this->testname;
  name += "_diff_q.vtu";
  std::ofstream output(name);
  data_out.write_vtu(output);
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
void CompareTest<dim, qdim>::GetL2ResidualsEigen(
    std::vector<double> &l2_residuals,
    const std::vector<pgd::sn::Cache> &caches,
    const std::vector<dealii::Vector<double>> &modes_energy,
    const pgd::sn::Transport<dim, qdim> &transport,
    const MomentToDiscrete<qdim> &m2d,
    const FixedSourceProblem<dim, qdim> &problem,
    const std::vector<double> &eigenvalues,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = mgxs->total.size();
  dealii::Vector<double> scattered(quadrature.size()*dof_handler.n_dofs());
  dealii::BlockVector<double> residual(num_groups, 
                                       quadrature.size()*dof_handler.n_dofs());
  dealii::BlockVector<double> flux(residual);
  dealii::BlockVector<double> residual_g(
      quadrature.size(), dof_handler.n_dofs());
  dealii::Vector<double> residual_l2(dof_handler.n_dofs());
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  dealii::Vector<double> streamed_k(dof_indices.size());
  dealii::Vector<double> mass_inv_streamed_k(dof_indices.size());
  std::vector<dealii::BlockVector<double>> boundary_conditions;
  for (int m = 0; m < l2_residuals.size(); ++m) {
    // get norm of residual
    double l2_flux = 0;
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
      residual_g = flux.block(g);
      for (int n = 0; n < quadrature.size(); ++n) {
        transport.collide_ordinate(residual_l2, residual_g.block(n));
        l2_flux += (residual_g.block(n) * residual_l2) 
                   * quadrature.weight(n) / width;
      }
    }
    l2_flux = std::sqrt(l2_flux);
    l2_residuals[m] = m ? (std::sqrt(l2_residuals[m])/(eigenvalues[m-1]*l2_flux)) 
                        : std::nan("z");
    table.add_value(key, l2_residuals[m]);
    if (m == l2_residuals.size()-1)
      continue;
    // update flux
    for (int g = 0; g < num_groups; ++g) {
      flux.block(g).add(modes_energy[m][g], caches[m].mode.block(0));
    }
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
            dof_m *= -eigenvalues[m];
            for (int gp = 0; gp < num_groups; ++gp)
              dof_m -= mgxs->chi[g][j] * mgxs->nu_fission[gp][j] 
                        * scattered[ni] * modes_energy[m][gp];
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
void CompareTest<dim, qdim>::GetL2ResidualsEigenMoments(
    std::vector<double> &l2_residuals,
    const std::vector<pgd::sn::Cache> &caches,
    const std::vector<dealii::Vector<double>> &modes_energy,
    const pgd::sn::Transport<dim, qdim> &transport,
    const DiscreteToMoment<qdim> &d2m,
    const FixedSourceProblem<dim, qdim> &problem,
    const std::vector<double> &eigenvalues,
    dealii::ConvergenceTable &table,
    const std::string &key) {
  const int num_groups = mgxs->total.size();
  dealii::BlockVector<double> residual(num_groups, dof_handler.n_dofs());
  dealii::BlockVector<double> flux(residual);
  dealii::Vector<double> residual_l2(dof_handler.n_dofs());
  dealii::BlockVector<double> streamed_d(
      quadrature.size(), dof_handler.n_dofs());
  dealii::BlockVector<double> streamed_m(1, dof_handler.n_dofs());
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  std::vector<dealii::BlockVector<double>> boundary_conditions;
  for (int m = 0; m < l2_residuals.size(); ++m) {
    // get norm of residual
    double l2_flux = 0;
    for (int g = 0; g < num_groups; ++g) {
      int g_rev = num_groups - 1 - g;
      double width =
          std::log(mgxs->group_structure[g_rev+1]
                    /mgxs->group_structure[g_rev]);
      AssertThrow(width > 0, dealii::ExcInvalidState());
      transport.collide_ordinate(residual_l2, residual.block(g));
      l2_residuals[m] += (residual.block(g) * residual_l2) / width;
      transport.collide_ordinate(residual_l2, flux.block(g));
      l2_flux += (flux.block(g) * residual_l2)  / width;
    }
    l2_flux = std::sqrt(l2_flux);
    l2_residuals[m] = m ? (std::sqrt(l2_residuals[m]) /(eigenvalues[m-1]*l2_flux)) 
                        : std::nan("z");
    table.add_value(key, l2_residuals[m]);
    if (m == l2_residuals.size()-1)
      continue;
    // update flux
    for (int g = 0; g < num_groups; ++g) {
      flux.block(g).add(modes_energy[m][g], caches[m].moments.block(0));
    }
    // update residual
    streamed_d = caches[m].streamed.block(0);
    transport.vmult_mass_inv(streamed_d);
    d2m.vmult(streamed_m, streamed_d);
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
        for (const dealii::types::global_dof_index i: dof_indices) {
          double dof_m = streamed_m.block(0)[i] * modes_energy[m][g];
          dof_m -= mgxs->total[g][j] * caches[m].moments.block(0)[i] 
                    * modes_energy[m][g];
          for (int gp = 0; gp < num_groups; ++gp)
            dof_m += mgxs->scatter[g][gp][j] * caches[m].moments.block(0)[i]
                      * modes_energy[m][gp];
          dof_m *= -eigenvalues[m];
          for (int gp = 0; gp < num_groups; ++gp)
            dof_m -= mgxs->chi[g][j] * mgxs->nu_fission[gp][j] 
                      * caches[m].moments.block(0)[i] * modes_energy[m][gp];
          residual.block(g)[i] -= dof_m;
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
    const Transport<dim, qdim> &transport,
    const dealii::BlockVector<double> *adjoint,
    std::vector<dealii::Vector<double>> *adjoints_energy,
    std::vector<dealii::BlockVector<double>> *adjoints_spaceangle) {
  AssertDimension(svecs_spaceangle.size(), 0);
  AssertDimension(svecs_energy.size(), 0);
  const int num_groups = flux.n_blocks();
  const int num_qdofs = flux.block(0).size();
  std::cout << "initialize flux matrix\n";
  dealii::LAPACKFullMatrix_<double> flux_matrix(num_qdofs, num_groups);
  dealii::LAPACKFullMatrix_<double> adjoint_matrix;
  if (adjoint != nullptr) {
    std::cout << adjoint->size() << "\n";
    std::cout << num_qdofs * num_groups << "\n";
    adjoint_matrix.reinit(num_qdofs, num_groups);
  }
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
            if (adjoint != nullptr)
              adjoint_matrix(nc+i, g) += mass_cho[i][j] 
                        * adjoint->block(g)[nc+j]
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
  if (adjoints_energy != nullptr) {
    // do decomposition of adjoint to get energy test functions
    // flux_matrix = U S V^T
    // adjoint_matrix = U S X^T --> X^T = (U S)^-1 adjoint_matrix
    // X^T = S^-1 U^-1 adjoint_matrix, where U^-1 = U^T
    dealii::LAPACKFullMatrix_<double> adjoint_energy(num_svecs, num_groups);
    adjoints_energy->resize(num_svecs, dealii::Vector<double>(num_groups));
    // dealii::LAPACKFullMatrix_<double> u_s;//(flux_matrix.get_svd_u());
    // dealii::LAPACKFullMatrix_<double> pseudoinverse(num_qdofs, num_qdofs);
    // u_s = flux_matrix.get_svd_u();
    // for (int s = 0; s < num_svecs; ++s)
    //   for (int ni = 0; ni < num_qdofs; ++ni) 
    //     u_s(ni, s) *= flux_matrix.singular_value(s);
    // std::cout << "assembled u_s\n";
    // u_s.invert();
    // std::cout << "inverted u_s\n";
    // u_s.mmult(adjoint_energy, adjoint_matrix);
    // std::cout << "mmult u_s\n";
    flux_matrix.get_svd_u().Tmmult(adjoint_energy, adjoint_matrix);
    for (int s = 0; s < num_svecs; ++s) {
      for (int g = 0; g < num_groups; ++g) {
        int g_rev = num_groups - 1 - g;
        double lower = mgxs->group_structure[g_rev] > 0 ? 
                        mgxs->group_structure[g_rev] : lowest;
        double width =
            std::log(mgxs->group_structure[g_rev+1]/lower);
        (*adjoints_energy)[s][g] = adjoint_energy(s, g) * std::sqrt(width);
      }
      (*adjoints_energy)[s] /= flux_matrix.singular_value(s);
    }
    // print out test functions
    std::ofstream adjoint_file;
    adjoint_file.open("adjoint_test_functions.txt");
    for (int s = 0; s < num_svecs; ++s) {
      for (int g = 0; g < num_groups; ++g) {
        adjoint_file << (*adjoints_energy)[s][g];
        if (g < num_groups - 1)
          adjoint_file << " ";
      }
      adjoint_file << "\n";
    }
    adjoint_file.close();
  }
  if (adjoints_spaceangle != nullptr) {
    // do decomposition of adjoint to get spatio-angular test functions
    // flux_matrix = U S V^T
    // adjoint_matrix = U S X^T
    // adjoint_matrix^T = X S U^T
    // U^T = (X S)^-1 adjoint_matrix^T
    // U^T = S^-1 X^-1 adjoint_matrix^T
    // U = adjoint_matrix X S^-1
    // X^T = S^-1 U^-1 adjoint_matrix, where U^-1 = U^T
    dealii::LAPACKFullMatrix_<double> adjoint_spaceangle(num_qdofs, num_svecs);
    adjoints_spaceangle->resize(num_svecs, 
        dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
    adjoint_matrix.mTmult(adjoint_spaceangle, flux_matrix.get_svd_vt());
    for (int s = 0; s < num_svecs; ++s) {
      for (int n = 0; n < quadrature.size(); ++n) {
        for (int c = 0; c < masses_cho.size(); ++c) {
          int nc = n * dof_handler.n_dofs() 
                    + c * dof_handler.get_fe().n_dofs_per_cell();
          dealii::FullMatrix<double> mass_cho_inv = masses_cho[c];
          for (int i = 0; i < mass_cho_inv.m(); ++i) {
            for (int j = 0; j < mass_cho_inv.n(); ++j) {
              (*adjoints_spaceangle)[s][nc+i] += 
                  mass_cho_inv[i][j]  * adjoint_spaceangle(nc+j, s)
                  / std::sqrt(quadrature.weight(n));
            }
          }
        }
      }
      (*adjoints_spaceangle)[s] /= flux_matrix.singular_value(s);
    }
    std::cout << "computed adjoints spaceangle\n";
  }
}

template <int dim, int qdim>
void CompareTest<dim, qdim>::Compare(int num_modes,
                                     const int max_iters_nonlinear,
                                     const double tol_nonlinear,
                                     const int max_iters_fullorder,
                                     const double tol_fullorder,
                                     const bool do_update,
                                     const bool do_minimax,
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
  bool do_adjoint = false;
  if (do_adjoint) {
    problem_full.fixed_source.transposed = true;
    problem_full.fission.transposed = true;
  }
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
    filename_full = filebase + "_k_full" + (do_adjoint ? "_adj" : "") + ".h5";
  bool do_forward_adjoint = true;
  bool do_lorenzi = do_forward_adjoint && false;
  std::string filename_adjoint = 
      filebase + "_k_full" + (do_adjoint ? "_adj" : "") + ".h5";
  dealii::BlockVector<double> flux_full_adjoint;
  namespace HDF5 = dealii::HDF5;
  if (precomputed_full) {
    HDF5::File file(filename_full, HDF5::File::FileAccessMode::open);
    flux_full = file.open_dataset("flux_full").read<dealii::Vector<double>>();
    if (do_forward_adjoint) {
      HDF5::File file_adjoint(filename_adjoint, 
                              HDF5::File::FileAccessMode::open);
      flux_full_adjoint.reinit(flux_full.get_block_indices());
      flux_full_adjoint = 
          file_adjoint.open_dataset("flux_full").read<dealii::Vector<double>>();
    }
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
  double sum = 0;
  for (int i = 0; i < flux_full.size(); ++i)
    sum += flux_full[i];
  if (sum < 0)
    flux_full *= -1;
  this->PlotFlux(flux_full, problem_full.d2m, mgxs->group_structure, "full_adj");
  int lower_left = 0;
  bool found = false;
  for (auto cell = dof_handler.begin_active(); 
       cell != dof_handler.end() && !found; ++cell) {
    for (int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; 
         ++v, ++lower_left) {
      std::cout << cell->vertex(v) << "\n";
      if (cell->vertex(v)[0] == 0 && cell->vertex(v)[1] == 0) {
        found = true;
        std::vector<dealii::types::global_dof_index> dof_indices(
            dof_handler.get_fe().dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        std::cout << dof_indices[v] << "\n";
        break;
      }
    }
  }
  // std::cout << (++mesh.begin_active())->vertex(1) << "\n";
  std::cout << lower_left << "\n";
  std::div_t wrap = 
      std::div(lower_left, dealii::GeometryInfo<dim>::vertices_per_cell);
  int i = 0;
  auto cell = mesh.begin_active();
  for (; i < wrap.quot; ++cell, ++i) {}
  std::cout << cell->vertex(wrap.rem) << "\n";
  std::ofstream spectrum_file;
  spectrum_file.open(
      filebase + "_spectrum" + (do_adjoint ? "_adj" : "") + ".txt");
  for (int g = 0; g < num_groups; ++g) {
    double spectrum = 0;
    for (int n = 0; n < quadrature.size(); ++n) {
      int gni = g*quadrature.size()*dof_handler.n_dofs() + 
                n*dof_handler.n_dofs() + lower_left;
      spectrum += quadrature.weight(n) * flux_full[gni];
    }
    spectrum_file << spectrum << "\n";
  }
  spectrum_file.close();
  for (int n = 0; n < quadrature.size(); ++n)
  if (full_only)
    return;
  // Compute svd of full order
  std::cout << "compute svd\n";
  std::vector<dealii::BlockVector<double>> svecs_spaceangle;
  std::vector<dealii::Vector<double>> svecs_energy;
  std::vector<dealii::Vector<double>> adjoint_energy;
  std::vector<dealii::BlockVector<double>> adjoint_spaceangle;
  ComputeSvd(svecs_spaceangle, svecs_energy, flux_full, 
              problem_full.transport, 
              do_lorenzi ? &flux_full_adjoint : nullptr,
              do_lorenzi ? &adjoint_energy : nullptr,
              do_lorenzi ? &adjoint_spaceangle : nullptr);
  if (do_forward_adjoint && !do_lorenzi) {
    ComputeSvd(adjoint_spaceangle, adjoint_energy, flux_full_adjoint,
               problem_full.transport);
  }
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
  pgd::sn::EnergyMgFull& energy_op = 
      do_eigenvalue ? energy_fiss : energy_mg;
  pgd::sn::FixedSourceP<dim>& spatioangular_op = 
      do_eigenvalue ? fission_source_p : fixed_source_p;
  const std::string filename_pgd = filebase + "_pgd_" +
      (do_update ? "update" : "prog") +
      (do_minimax ? "_minimax" : "") +
      (do_eigenvalue ? "_k" : "") + ".h5";
  std::vector<double> eigenvalues;
  if (precomputed_pgd) {
    // read from file
    HDF5::File file(filename_pgd, HDF5::File::FileAccessMode::open);
    if (do_eigenvalue) {
      eigenvalues = 
          file.open_dataset("eigenvalues").read<std::vector<double>>();
    }
    for (int m = 0; m < num_modes; ++m) {
      const std::string mm = std::to_string(m);
      spatioangular_op.enrich(0);
      spatioangular_op.caches.back().mode = file.open_dataset(
          "modes_spaceangle"+mm).read<dealii::Vector<double>>();
      spatioangular_op.set_cache(spatioangular_op.caches.back());
      energy_op.modes.push_back(file.open_dataset(
          "modes_energy"+mm).read<dealii::Vector<double>>());
    }
    std::cout <<"read precomputed pgd\n";
  } else {
    // run pgd
    std::vector<pgd::sn::LinearInterface*> linear_ops = 
        {&energy_op, &spatioangular_op};
    std::unique_ptr<pgd::sn::NonlinearGS> nonlinear_gs;
    energy_op.do_minimax = do_minimax;
    spatioangular_op.do_minimax = do_minimax;
    if (do_eigenvalue) {
      nonlinear_gs =
          std::make_unique<pgd::sn::EigenGS>(linear_ops, num_materials, 1);
      auto eigen_gs = dynamic_cast<pgd::sn::EigenGS*>(nonlinear_gs.get());
      double k0 = eigen_gs->initialize_guess();
      std::cout << "initial k (guess): " << k0 << "\n";
    } else {
      nonlinear_gs = std::make_unique<pgd::sn::NonlinearGS>(
          linear_ops, num_materials, 1, num_sources);
    }
    std::vector<int> unconverged;
    std::vector<double> residuals;
    std::cout << "run pgd\n";
    RunPgd(*nonlinear_gs, num_modes, max_iters_nonlinear, tol_nonlinear,
            do_update, unconverged, residuals, 
            do_eigenvalue ? &eigenvalues : nullptr);
    std::cout << "done running pgd\n";
    if (do_eigenvalue) {
      // normalize by l2 norm of expanded flux
      dealii::BlockVector<double> flux_pgd(flux_full);
      flux_pgd = 0;
      for (int m = 0; m < num_modes; ++m) {
        for (int g = 0; g < num_groups; ++g) {
          flux_pgd.block(g).add(energy_op.modes[m][g], 
                                spatioangular_op.caches[m].mode.block(0));
        }
      }
      double norm_expanded = flux_pgd.l2_norm();
      for (int m = 0; m < num_modes; ++m)
        energy_op.modes[m] /= norm_expanded;
      sum = 0;
      for (int i = 0; i < flux_pgd.size(); ++i)
        sum += flux_pgd[i];
      if (sum < 0)
        for (int m = 0; m < num_modes; ++m)
          energy_op.modes[m] *= -1;
    }
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
      file.write_dataset("modes_spaceangle"+mm, 
          spatioangular_op.caches[m].mode.block(0));
      file.write_dataset("modes_energy"+mm, energy_op.modes[m]);
    }
    if (do_eigenvalue) {
      file.write_dataset("eigenvalues", eigenvalues);
    }
  }
  std::vector<dealii::BlockVector<double>> modes_spaceangle(num_modes,
      dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
  for (int m = 0; m < num_modes; ++m) {
    modes_spaceangle[m] = spatioangular_op.caches[m].mode.block(0);
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
    const int jfnk_iters = 50;
    // set adjoint modes
    if (do_forward_adjoint) {
      for (int s = 0; s < num_svecs; ++s) {
        energy_fiss.modes_adj.push_back(adjoint_energy[s]);
        subspace_problem.fixed_source_s.test_functions.emplace_back(
            adjoint_spaceangle[s].size());
        subspace_problem.fixed_source_s.test_functions[s] = 
            adjoint_spaceangle[s];
      }
    }
    energy_fiss.do_minimax = false;
    for (int i = 0; i <= jfnk_iters; ++i) {
      std::cout << "setting modes " << modes_all.l2_norm() << "\n";
      modes_all.block(0) /= modes_all.block(0).l2_norm();
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
      if (false) {  // do spatio-angular eigensolve
        modes = modes_all.block(0);
        subspace_problem.set_cross_sections(
          jacobian.inner_products_unperturbed[1]);
        // std::cout << "ax, bx: " << ax.l2_norm() << ", " << bx.l2_norm() << "\n";
        // ax.add(-rayleigh, bx);
        // std::cout << "initial residual: " << ax.l2_norm() << std::endl;
        // dealii::BlockVector<double> ax_mass_inv(ax);
        // for (int m = 0; m < ax_mass_inv.n_blocks(); ++m)
        //   problem.transport.vmult_mass_inv(ax_mass_inv.block(m));
        // std::cout << "initial residual: " << ax_mass_inv.l2_norm() << std::endl;
        // bx = 0;
        // subspace_problem.fixed_source_s_gs.vmult(bx, ax);
        // std::cout << "fixed gs residual: " << bx.l2_norm() << std::endl;
        // bx = 0;
        // subspace_problem.fission_s_gs.vmult(bx, ax);
        // std::cout << "fission gs residual: " << bx.l2_norm() << std::endl;
        subspace_problem.fission_s_gs.set_shift(k_energy);
        const int num_qdofs = quadrature.size() * dof_handler.n_dofs();
        aether::pgd::sn::FissionSourceShiftedS<dim> fission_source_s(
            subspace_problem.fission_s,
            subspace_problem.fixed_source_s,
            subspace_problem.fixed_source_s_gs);
        // fission_source_s.shift = 0;
        // dealii::BlockVector<double> modes_power(modes);
        // fission_source_s.vmult(modes_power, modes);
        // modes_power /= modes_power.l2_norm();
        // modes_all.block(0) = modes_power;
        // continue;
        ::aether::PETScWrappers::BlockWrapper fission_source_s_petsc(
            num_modes_s, MPI_COMM_WORLD, num_qdofs, num_qdofs,
            fission_source_s);

        ::aether::PETScWrappers::BlockWrapper fixed_source_s(
            num_modes_s, MPI_COMM_WORLD, num_qdofs, num_qdofs,
            subspace_problem.fixed_source_s);
        // ::aether::PETScWrappers::BlockWrapper fission_s_gs(
        //     num_modes_s, MPI_COMM_WORLD, num_qdofs, num_qdofs,
        //     subspace_problem.fission_s_gs);
        ::aether::PETScWrappers::BlockWrapper fission_s(
            num_modes_s, MPI_COMM_WORLD, num_qdofs, num_qdofs, 
            subspace_problem.fission_s);
        const int size = num_modes_s * num_qdofs;
        std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
        eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
        eigenvectors[0].compress(dealii::VectorOperation::insert);
        const double norm = 1; //modes.l2_norm();
        for (int i = 0; i < size; ++i)
          eigenvectors[0][i] = modes[i] / norm;
        // for (int i = 0; i < num_qdofs; ++i)
        //   eigenvectors[0][i] = 1;
        eigenvectors[0].compress(dealii::VectorOperation::insert);
        std::vector<double> eigenvalues = {k_energy};
        dealii::SolverControl control(20, std::max(1e-6, std::pow(10, -2-i)));
        // dealii::SolverControl control(100, 1e-6);
        dealii::SLEPcWrappers::SolverKrylovSchur eigensolver(control);
        // dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
        eigensolver.set_initial_space(eigenvectors);
        eigensolver.set_target_eigenvalue(k_energy);
        eigensolver.set_which_eigenpairs(EPS_LARGEST_MAGNITUDE);
        // shift and invert
        using Shift 
            = dealii::SLEPcWrappers::TransformationShift::AdditionalData;
        dealii::SLEPcWrappers::TransformationShift shift(
            MPI_COMM_WORLD, Shift(k_energy));
        shift.set_matrix_mode(ST_MATMODE_SHELL);
        // eigensolver.set_transformation(shift);
        /*
        // dealii::IterationNumberControl control_inv(10, 1e-4);
        // dealii::PETScWrappers::SolverGMRES solver_inv(control_inv, MPI_COMM_WORLD);
        // dealii::PETScWrappers::PreconditionNone preconditioner(fixed_source_s);
        // aether::PETScWrappers::PreconditionerMatrix preconditioner(fixed_source_s_gs);
        aether::PETScWrappers::PreconditionerShell preconditioner(fission_s_gs);
        // solver_inv.initialize(preconditioner);
        // std::cout << "b: "; preconditioner.vmult(bar, foo);
        // shift_invert.set_solver(solver_inv);
        // std::cout << "c: "; preconditioner.vmult(bar, foo);
        // eigensolver.set_transformation(shift_invert);
        // std::cout << "d: "; preconditioner.vmult(bar, foo);
        dealii::SolverControl control_dummy(1, 0);
        dealii::PETScWrappers::SolverPreOnly solver_pc(control_dummy);
        solver_pc.initialize(preconditioner);
        // dealii::IterationNumberControl control_si(3, 1e-6);
        // dealii::PETScWrappers::SolverGMRES solver_si(control_si);
        // solver_si.initialize(preconditioner);
        // shift.set_solver(solver_si);
        // eigensolver.set_transformation(shift);
        aether::SLEPcWrappers::TransformationPreconditioner stprecond(
            MPI_COMM_WORLD, fission_s_gs);
        stprecond.set_matrix_mode(ST_MATMODE_SHELL);
        stprecond.set_solver(solver_pc);
        eigensolver.set_transformation(stprecond);
        */
        try {
          std::cout << "to solve\n";
          eigensolver.solve(fission_source_s_petsc, eigenvalues, eigenvectors);
          // eigensolver.solve(fission_s, fixed_source_s, eigenvalues, eigenvectors);
          for (int i = 0; i < eigenvectors[0].size(); ++i)
            modes[i] = eigenvectors[0][i];
        } catch (dealii::SolverControl::NoConvergence &failure) {
          failure.print_info(std::cout);
        } //catch (...) {}
        std::cout << "SPATIO-ANGULAR EIGENVALUE: " << eigenvalues[0] << std::endl;
        if (eigenvalues[0] < 0) {
          // eigenvalues[0] *= -1;
          // eigenvectors[0] *= -1;
        }
        modes_all.block(0) = eigenvectors[0];
        continue;
      }
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
          // modes.add(-k_energy, modes_copy);
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
    if (!energy_fiss.modes_adj.empty()) {
      energy_mg.modes_adj = energy_fiss.modes_adj;
    }
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
    file.set_attribute("num_modes", num_modes_s);
    for (int m = 0; m < num_modes; ++m) {
      const std::string mm = std::to_string(m);
      file.write_dataset("modes_spaceangle"+mm, modes.block(m));
      file.write_dataset("modes_energy"+mm, energy_mg.modes[m]);
      if (!energy_mg.modes_adj.empty())
        file.write_dataset("modes_energy_adj"+mm, energy_mg.modes_adj[m]);
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
  GetL2ErrorsDiscrete(l2_errors_d, modes_spaceangle, energy_op.modes, 
                      flux_full, problem.transport, table, "error_d");
  GetL2ErrorsMoments(l2_errors_m, modes_spaceangle, energy_op.modes, 
                      flux_full, problem.transport, problem.d2m, table, 
                      "error_m");
  GetL2Norms(l2_norms, modes_spaceangle, energy_op.modes, problem.transport,
              table, "norm");
  if (!do_eigenvalue) {
  std::cout << "residuals\n";
  GetL2Residuals(l2_residuals, spatioangular_op.caches, energy_op.modes, 
                  source_full, problem.transport, problem.m2d, problem_full,
                  false, table, "residual");
  std::cout << "residuals streamed\n";
  GetL2Residuals(l2_residuals_streamed, spatioangular_op.caches, 
                  energy_op.modes, source_full, problem.transport, problem.m2d,
                  problem_full, true, table, "residual_streamed");
  dealii::BlockVector<double> uncollided(source_full.get_block_indices());
  problem_full.sweep_source(uncollided, source_full);
  GetL2ResidualsFull(l2_residuals_swept, modes_spaceangle, energy_op.modes, 
                      uncollided, problem.transport, problem_full, table, 
                      "residual_swept");
  } else {
    std::vector<double> l2_errors_q(num_modes+1);
    GetL2ErrorsFissionSource(l2_errors_q, modes_spaceangle, energy_op.modes, 
                             flux_full, problem.transport, problem.d2m, 
                             problem_full.production, table, "error_q");
    GetL2ResidualsEigen(l2_residuals, spatioangular_op.caches, energy_op.modes, 
                        problem.transport, problem.m2d, problem_full, 
                        eigenvalues, table, "residual");
    table.add_value("error_k", std::nan("k"));
    for (int m = 0; m < num_modes; ++m) {
      table.add_value("error_k", 1e5 * (eigenvalues[m] - eigenvalue_full));
    }
    table.set_scientific("error_k", true);
    table.set_precision("error_k", 16);
  }
  if (num_modes_s > 0)
    this->WriteConvergenceTable(table, "_M"+std::to_string(num_modes_s));
  else
    this->WriteConvergenceTable(table);
}

template class CompareTest<2>;
