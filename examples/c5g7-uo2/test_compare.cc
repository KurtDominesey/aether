#include "test_harness.cc"

namespace {

class C5G7CompareTest : public C5G7Test {
 protected:
  void SetUp() override {
    C5G7Test::SetUp();
    mesh.refine_global(2);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
  }

  void WriteUniformFissionSource(
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
  void RunFullOrder(dealii::BlockVector<double> &flux,
                    const dealii::BlockVector<double> &source,
                    const FixedSourceProblem<dim, qdim> &problem,
                    const int max_iters, const double tol) {
    dealii::BlockVector<double> uncollided(source.get_block_indices());
    problem.sweep_source(uncollided, source);
    dealii::SolverControl control(max_iters, tol);
    dealii::PreconditionIdentity preconditioner;
    dealii::SolverRichardson<dealii::BlockVector<double>> solver(control);
    solver.solve(problem.fixed_source, flux, uncollided, preconditioner);
  }

  void RunPgd(pgd::sn::NonlinearGS &nonlinear_gs, const int num_modes,
              const int max_iters, const double tol, const bool do_update) {
    dealii::BlockVector<double> _;
    for (int m = 0; m < num_modes; ++m) {
      nonlinear_gs.enrich();
      for (int k = 0; k < max_iters; ++k) {
        double residual = nonlinear_gs.step(_, _);
        if (residual < tol)
          break;
      }
      if (do_update) {
        nonlinear_gs.finalize();
        if (m > 0)
          nonlinear_gs.update();
      }
    }
  }

  template <int dim, int qdim>
  void GetL2ErrorsDiscrete(
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
      for (int m = 0; m < l2_errors.size(); ++m) {
        if (m > 0)
          modal[g].add(modes_energy[m-1][g], modes_spaceangle[m-1]);
        for (int n = 0; n < quadrature.size(); ++n) {
          diff = modal[g].block(n);
          diff -= reference_g.block(n);
          transport.collide(diff_l2, diff);
          l2_errors[m] += quadrature.weight(n) * (diff * diff_l2);
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
  void GetL2ErrorsMoments(
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
      for (int m = 0; m < l2_errors.size(); ++m) {
        if (m > 0) {
          mode.equ(modes_energy[m-1][g], modes_spaceangle[m-1]);
          d2m.vmult_add(modal[g], mode);
        }
        diff = modal[g].block(0);
        diff -= reference_g_m.block(0);
        transport.collide(diff_l2, diff);
        l2_errors[m] += diff * diff_l2;
      }
    }
    for (int m = 0; m < l2_errors.size(); ++m) {
      l2_errors[m] = std::sqrt(l2_errors[m]);
      table.add_value(key, l2_errors[m]);
    }
    table.set_scientific(key, true);
    table.set_precision(key, 16);
  }

  void GetL2Norms(
      std::vector<double> &l2_norms,
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const pgd::sn::Transport<dim, qdim> &transport,
      dealii::ConvergenceTable &table,
      const std::string &key) {
    const int num_groups = mgxs->total.size();
    dealii::Vector<double> mode_l2(modes_spaceangle[0].block(0).size());
    for (int m = 0; m < l2_norms.size() - 1; ++m) {
      double l2_energy = modes_energy[m].l2_norm();
      double l2_spaceangle = 0;
      for (int n = 0; n < quadrature.size(); ++n) {
        transport.collide(mode_l2, modes_spaceangle[m].block(n));
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

  void GetL2Residuals(
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
      // for (int sweep = 0; sweep < 100; ++sweep)
      // problem.sweep_source(swept, residual);
      for (int g = 0; g < num_groups; ++g) {
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
            transport.collide(residual_l2, residual_g.block(n));
            l2_residuals[m] += (residual_g.block(n) * residual_l2) 
                              * quadrature.weight(n);
          } else {
            transport.collide(swept_l2, swept_g.block(n));
            l2_residuals[m] += (swept_g.block(n) * swept_l2) 
                              * quadrature.weight(n);
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
          for (int n = 0; n < quadrature.size(); ++n) {
            for (int i = 0; i < dof_indices.size(); ++i) {
              const dealii::types::global_dof_index ni =
                  n * dof_handler.n_dofs() + dof_indices[i];
              streamed_k[i] = caches[m].streamed.block(0)[ni];
            }
            dealii::FullMatrix<double> mass = transport.cell_matrices[c].mass;
            mass.gauss_jordan();
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

  void GetL2ResidualsFull(
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
        residual_g = residual.block(g);
        for (int n = 0; n < quadrature.size(); ++n) {
          transport.collide(residual_l2, residual_g.block(n));
          l2_residuals[m] += (residual_g.block(n) * residual_l2)
                             * quadrature.weight(n);
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
};

TEST_F(C5G7CompareTest, Compare) {
  const int num_groups = mgxs->total.size();
  const int num_materials = mgxs->total[0].size();
  // Create sources
  std::vector<dealii::Vector<double>> sources_energy;
  std::vector<dealii::BlockVector<double>> sources_spaceangle;
  WriteUniformFissionSource(sources_energy, sources_spaceangle);
  const int num_sources = sources_energy.size();
  // Create boundary conditions
  std::vector<dealii::BlockVector<double>> boundary_conditions_one;
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
  FixedSourceProblem<dim, qdim> problem_full(
      dof_handler, quadrature, *mgxs, boundary_conditions);
  RunFullOrder(flux_full, source_full, problem_full, 1000, 1e-6);
  // Run pgd model
  Mgxs mgxs_one(1, num_materials, 1);
  Mgxs mgxs_pseudo(1, num_materials, 1);
  for (int j = 0; j < num_materials; ++j) {
    mgxs_one.total[0][j] = 1;
    mgxs_one.scatter[0][0][j] = 1;
  }
  using TransportType = pgd::sn::Transport<dim, qdim>;
  using TransportBlockType = pgd::sn::TransportBlock<dim, qdim>;
  FixedSourceProblem<dim, qdim, TransportType, TransportBlockType> problem(
      dof_handler, quadrature, mgxs_pseudo, boundary_conditions);
  pgd::sn::FixedSourceP fixed_source_p(
      problem.fixed_source, mgxs_pseudo, mgxs_one, sources_spaceangle);
  pgd::sn::EnergyMgFull energy_mg(*mgxs, sources_energy);
  std::vector<pgd::sn::LinearInterface*> linear_ops = 
      {&fixed_source_p, &energy_mg};
  pgd::sn::NonlinearGS nonlinear_gs(linear_ops, num_materials, 1, num_sources);
  const int num_modes = 50;
  RunPgd(nonlinear_gs, num_modes, 50, 1e-6, false);
  std::cout << "done running pgd\n";
  std::vector<dealii::BlockVector<double>> modes_spaceangle(num_modes,
      dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
  for (int m = 0; m < num_modes; ++m)
    modes_spaceangle[m] = fixed_source_p.caches[m].mode.block(0);
  dealii::ConvergenceTable table;
  std::vector<double> l2_errors_d(num_modes+1);
  std::vector<double> l2_errors_m(num_modes+1);
  std::vector<double> l2_residuals(num_modes+1);
  std::vector<double> l2_residuals_streamed(num_modes+1);
  std::vector<double> l2_residuals_swept(num_modes+1);
  std::vector<double> l2_norms(num_modes+1);
  std::cout << "get l2 errors discrete\n";
  GetL2ErrorsDiscrete(l2_errors_d, modes_spaceangle, energy_mg.modes, flux_full, 
                      problem.transport, table, "error_d");
  GetL2ErrorsMoments(l2_errors_m, modes_spaceangle, energy_mg.modes, flux_full, 
                     problem.transport, problem.d2m, table, "error_m");  
  GetL2Residuals(l2_residuals, fixed_source_p.caches, energy_mg.modes, 
                 source_full, problem.transport, problem.m2d, problem_full,
                 false, table, "residual");
  GetL2Residuals(l2_residuals_streamed, fixed_source_p.caches, energy_mg.modes, 
                 source_full, problem.transport, problem.m2d, problem_full,
                 true, table, "residual_streamed");
  GetL2Norms(l2_norms, modes_spaceangle, energy_mg.modes, problem.transport,
             table, "norm");
  dealii::BlockVector<double> uncollided(source_full.get_block_indices());
  problem_full.sweep_source(uncollided, source_full);
  GetL2ResidualsFull(l2_residuals_swept, modes_spaceangle, energy_mg.modes, 
                     uncollided, problem.transport, problem_full, table, 
                     "residual_swept");
  WriteConvergenceTable(table);
}

}