#include "harness_test.cc"

class C5G7MmsOrderTest : public C5G7Test, 
                         public ::testing::WithParamInterface<int> {
 protected:
  void SetUp() override {
    C5G7Test::SetUp();
    dealii::FE_DGQ<dim> fe(this->GetParam());
    this->dof_handler.initialize(this->mesh, fe);
  }
};

TEST_P(C5G7MmsOrderTest, FullOrder) {
  const int num_groups = this->mgxs->total.size();
  const int num_materials = this->mgxs->total[0].size();
  std::vector<std::vector<dealii::BlockVector<double>>> boundary_conditions(
      num_groups, std::vector<dealii::BlockVector<double>>());
  // Create solution
  double factor = dealii::numbers::PI_2 / this->pitch;
  using Solution = CosineFunction<dim>;
  std::vector<Solution> solutions_spaceangle;
  solutions_spaceangle.emplace_back(2 * factor);
  std::vector<dealii::Vector<double>> solutions_energy(1,
      dealii::Vector<double>(num_groups));
  for (int g = 0; g < num_groups; ++g)
    solutions_energy[0][g] = g + 1;
  // Run MMS refinement cycles
  dealii::ConvergenceTable convergence_table;
  int num_cycles = 5;
  std::vector<double> l2_errors(num_cycles);
  std::vector<std::vector<double>> l2_errors_g(
      num_cycles, std::vector<double>(
          num_groups)
  );
  std::vector<std::vector<std::vector<double>>> l2_errors_n(
      num_cycles, std::vector<std::vector<double>>(
          num_groups, std::vector<double>(
              this->quadrature.size())
      )
  );
  for (int cycle = 0; cycle < num_cycles; ++cycle) {
    if (cycle > 0) {
      mesh.refine_global();
      dof_handler.initialize(mesh, dof_handler.get_fe());
    }
    // Define source
    dealii::BlockVector<double> source(
        num_groups, quadrature.size() * dof_handler.n_dofs());
    this->CreateSource(source, solutions_spaceangle, solutions_energy,
                       dof_handler, quadrature, *mgxs);
    // Run problem
    FixedSourceProblem<dim, qdim> problem(
        dof_handler, quadrature, *mgxs, boundary_conditions);
    dealii::BlockVector<double> uncollided(source.get_block_indices());
    dealii::BlockVector<double> flux(source.get_block_indices());
    problem.sweep_source(uncollided, source);
    dealii::SolverControl control(3000, 1e-6);
    dealii::SolverRichardson<dealii::BlockVector<double>> solver(control);
    dealii::PreconditionIdentity preconditioner;
    std::cout << "running full-order: cycle " << cycle << std::endl;
    solver.solve(problem.fixed_source, flux, uncollided, preconditioner);
    // Post-process
    dealii::BlockVector<double> flux_g(quadrature.size(), dof_handler.n_dofs());
    std::cout << "integrating error\n";
    for (int g = 0; g < num_groups; ++g) {
      flux_g = flux.block(g);
      flux_g /= solutions_energy[0][g];
      for (int n = 0; n < quadrature.size(); ++n) {
        dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
        dealii::VectorTools::integrate_difference(
            dof_handler, flux_g.block(n), solutions_spaceangle[0],
            difference_per_cell, 
            dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
            dealii::VectorTools::L2_norm);
        double l2_error_sq = 0;
        for (int i = 0; i < difference_per_cell.size(); ++i)
          l2_error_sq += std::pow(difference_per_cell[i], 2);
        l2_error_sq *= std::pow(solutions_energy[0][g], 2);
        double l2_error = std::sqrt(l2_error_sq);
        l2_errors_n[cycle][g][n] = l2_error;
        std::string gn_key = "g" + std::to_string(g) + "n" + std::to_string(n);
        convergence_table.add_value(gn_key, l2_error);
        convergence_table.set_scientific(gn_key, true);
        convergence_table.set_precision(gn_key, 16);
        double summand = l2_error_sq * quadrature.weight(n);
        l2_errors_g[cycle][g] += summand;
        l2_errors[cycle] += summand;
        if (cycle > 0) {
          double ratio = l2_errors_n[cycle-1][g][n] / l2_errors_n[cycle][g][n];
          double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
          EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 1e-1);
        }
      }
      l2_errors_g[cycle][g] = std::sqrt(l2_errors_g[cycle][g]);
      std::string g_key = "g" + std::to_string(g);
      convergence_table.add_value(g_key, l2_errors_g[cycle][g]);
      convergence_table.set_scientific(g_key, true);
      convergence_table.set_precision(g_key, 16);
    }
    l2_errors[cycle] = std::sqrt(l2_errors[cycle]);
    convergence_table.add_value("total", l2_errors[cycle]);
  }
  convergence_table.set_scientific("total", true);
  convergence_table.set_precision("total", 16);
  WriteConvergenceTable(convergence_table);
}

TEST_P(C5G7MmsOrderTest, Pgd) {
  const int num_groups = this->mgxs->total.size();
  const int num_materials = this->mgxs->total[0].size();
  std::vector<std::vector<dealii::BlockVector<double>>> boundary_conditions(
      1, std::vector<dealii::BlockVector<double>>());
  // Dummy mgxs for spatio-angular equation
  Mgxs mgxs_one(1, num_materials, 1);
  Mgxs mgxs_pseudo(1, num_materials, 1);
  for (int j = 0; j < num_materials; ++j) {
    mgxs_one.total[0][j] = 1;
    mgxs_one.scatter[0][0][j] = 1;
  }
  // Create solution
  double factor = dealii::numbers::PI_2 / this->pitch;
  using Solution = CosineFunction<dim>;
  std::vector<Solution> solutions_spaceangle;
  solutions_spaceangle.emplace_back(2 * factor);
  std::vector<dealii::Vector<double>> solutions_energy(1,
      dealii::Vector<double>(num_groups));
  for (int g = 0; g < num_groups; ++g)
    solutions_energy[0][g] = g + 1;
  // Run MMS refinement cycles
  dealii::ConvergenceTable convergence_table;
  int num_cycles = 5;
  std::vector<double> l2_errors(num_cycles);
  std::vector<std::vector<double>> l2_errors_g(
      num_cycles, std::vector<double>(
          num_groups)
  );
  std::vector<std::vector<std::vector<double>>> l2_errors_n(
      num_cycles, std::vector<std::vector<double>>(
          num_groups, std::vector<double>(
              this->quadrature.size())
      )
  );
  for (int cycle = 0; cycle < num_cycles; ++cycle) {
    if (cycle > 0) {
      mesh.refine_global();
      dof_handler.initialize(mesh, dof_handler.get_fe());
    }
    std::vector<dealii::BlockVector<double>> sources_spaceangle;
    std::vector<dealii::Vector<double>> sources_energy;
    CreateSources(sources_spaceangle, sources_energy,
                  solutions_spaceangle, solutions_energy,
                  dof_handler, quadrature, *mgxs);
    const int num_sources = sources_spaceangle.size();
    double factor = solutions_energy[0].l2_norm();
    for (int s = 0; s < num_sources; ++s) {
      sources_energy[s] /= factor;
      sources_spaceangle[s] *= factor;
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
    pgd::sn::NonlinearGS nonlinear_gs(
        linear_ops, num_materials, 1, num_sources);
    const double nonlinear_tol = 1e-6;
    const int num_modes = 1;
    for (int m = 0; m < num_modes; ++m) {
      nonlinear_gs.enrich();
      const int num_iters = 40;
      for (int k = 0; k < num_iters; ++k) {
        double residual = nonlinear_gs.step(dealii::BlockVector<double>(),
                                            dealii::BlockVector<double>());
        // if (residual < nonlinear_tol)
        //   break;
      }
      // nonlinear_gs.finalize();
      // if (m > 0)
      //   nonlinear_gs.update();
    }
    // Post-process
    dealii::BlockVector<double> flux_g(quadrature.size(), dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      flux_g = fixed_source_p.caches[0].mode.block(0);
      flux_g *= energy_mg.modes[0][g];
      for (int m = 1; m < num_modes; ++m) {
        // pass
      }
      flux_g /= solutions_energy[0][g];
      for (int n = 0; n < quadrature.size(); ++n) {
        dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
        dealii::VectorTools::integrate_difference(
            dof_handler, flux_g.block(n), solutions_spaceangle[0],
            difference_per_cell, 
            dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
            dealii::VectorTools::L2_norm);
        double l2_error_sq = 0;
        for (int i = 0; i < difference_per_cell.size(); ++i)
          l2_error_sq += std::pow(difference_per_cell[i], 2);
        l2_error_sq *= std::pow(solutions_energy[0][g], 2);
        double l2_error = std::sqrt(l2_error_sq);
        l2_errors_n[cycle][g][n] = l2_error;
        std::string gn_key = "g" + std::to_string(g) + "n" + std::to_string(n);
        convergence_table.add_value(gn_key, l2_error);
        convergence_table.set_scientific(gn_key, true);
        convergence_table.set_precision(gn_key, 16);
        double summand = l2_error_sq * quadrature.weight(n);
        l2_errors_g[cycle][g] += summand;
        l2_errors[cycle] += summand;
        if (cycle > 0) {
          double ratio = l2_errors_n[cycle-1][g][n] / l2_errors_n[cycle][g][n];
          double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
          EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 1e-1);
        }
      }
      l2_errors_g[cycle][g] = std::sqrt(l2_errors_g[cycle][g]);
      std::string g_key = "g" + std::to_string(g);
      convergence_table.add_value(g_key, l2_errors_g[cycle][g]);
      convergence_table.set_scientific(g_key, true);
      convergence_table.set_precision(g_key, 16);
    }
    l2_errors[cycle] = std::sqrt(l2_errors[cycle]);
    convergence_table.add_value("total", l2_errors[cycle]);
  }
  convergence_table.set_scientific("total", true);
  convergence_table.set_precision("total", 16);
  WriteConvergenceTable(convergence_table);
}

INSTANTIATE_TEST_CASE_P(FEDegree, C5G7MmsOrderTest, ::testing::Range(0, 3));