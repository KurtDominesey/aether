#ifndef AETHER_EXAMPLES_MGXS_TEST_H_
#define AETHER_EXAMPLES_MGXS_TEST_H_

#include "compare_test.h"

template <int dim, int qdim>
class MgxsTest : virtual public CompareTest<dim, qdim> {
 protected:
  using CompareTest<dim, qdim>::mesh;
  using CompareTest<dim, qdim>::quadrature;
  using CompareTest<dim, qdim>::dof_handler;
  using CompareTest<dim, qdim>::mgxs;


  void CompareMgxs(const int num_modes,
                   const int max_iters_nonlinear,
                   const double tol_nonlinear,
                   const int max_iters_fullorder,
                   const double tol_fullorder,
                   const bool do_update,
                   const std::vector<int> &g_maxes,
                   const std::vector<double> &volumes) {
    const int num_groups = mgxs->total.size();
    const int num_materials = mgxs->total[0].size();
    // Create sources
    std::vector<dealii::Vector<double>> sources_energy;
    std::vector<dealii::BlockVector<double>> sources_spaceangle;
    this->WriteUniformFissionSource(sources_energy, sources_spaceangle);
    const int num_sources = sources_energy.size();
    // Create boundary conditions
    std::vector<std::vector<dealii::BlockVector<double>>> 
        boundary_conditions_one(1);
    std::vector<std::vector<dealii::BlockVector<double>>> 
        boundary_conditions(num_groups);
    // Run full order
    std::cout << "run full order\n";
    dealii::BlockVector<double> flux_full(
        num_groups, quadrature.size()*dof_handler.n_dofs());
    dealii::BlockVector<double> source_full(flux_full.get_block_indices());
    for (int g = 0; g < num_groups; ++g)
      for (int j = 0; j < num_sources; ++j)
        source_full.block(g).add(
            sources_energy[j][g], sources_spaceangle[j].block(0));
    FixedSourceProblem<dim, qdim> problem_full(
        dof_handler, quadrature, *mgxs, boundary_conditions);
    this->RunFullOrder(flux_full, source_full, problem_full, 
                       max_iters_fullorder, tol_fullorder);
    dealii::BlockVector<double> flux_full_l0(num_groups, dof_handler.n_dofs());
    dealii::BlockVector<double> flux_full_l1(num_groups, 2*dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      problem_full.d2m.vmult(flux_full_l0.block(g), flux_full.block(g));
      std::cout << "discrete to legendre g=" << g << std::endl;
      problem_full.d2m.discrete_to_legendre(flux_full_l1.block(g), 
                                            flux_full.block(g));
    }
    std::cout << "collapsing spectra\n";
    std::vector<dealii::BlockVector<double>> spectra;
    collapse_spectra(spectra, flux_full_l0, dof_handler, 
                     problem_full.transport);
    std::cout << "collapsing mgxs\n";
    std::cout << spectra.size() << ", " << num_materials << std::endl
              << spectra[0].n_blocks() << ", " << spectra[0].block(0).size()
              << std::endl;
    Mgxs mgxs_coarse = collapse_mgxs(spectra, *mgxs, g_maxes);
    std::cout << "collapsed mgxs CP\n";
    Mgxs mgxs_coarse_ip = collapse_mgxs(
        flux_full_l1, dof_handler, problem_full.transport, *mgxs, g_maxes,
        INCONSISTENT_P);
    std::cout << "collapsed mgxs IP\n";
    // Compute svd
    std::cout << "compute svd\n";
    std::vector<dealii::BlockVector<double>> svecs_spaceangle;
    std::vector<dealii::Vector<double>> svecs_energy;
    // this->ComputeSvd(svecs_spaceangle, svecs_energy, flux_full, 
    //                  problem_full.transport);
    const int num_svecs = svecs_spaceangle.size();
    // Run infinite medium
    std::cout << "run infinite medium\n";
    dealii::Vector<double> spectrum(num_groups);
    RunInfiniteMedium(spectrum, sources_energy[2], *mgxs, volumes);
    Mgxs mgxs_coarse_inf = collapse_mgxs(spectrum, *mgxs, g_maxes);
    std::cout << "VOLUMES ";
    for (double volume : volumes)
      std::cout << volume << " ";
    std::cout << "\nSPECTRUM ";
    for (int g = 0; g < num_groups; ++g)
      std::cout << spectrum[g] << " ";
    std::cout << "\n";
    // Run pgd model
    std::cout << "run pgd model\n";
    Mgxs mgxs_one(1, num_materials, 1);
    Mgxs mgxs_pseudo(1, num_materials, 1);
    for (int j = 0; j < num_materials; ++j) {
      mgxs_one.total[0][j] = 1;
      mgxs_one.scatter[0][0][j] = 1;
    }
    using TransportType = pgd::sn::Transport<dim, qdim>;
    using TransportBlockType = pgd::sn::TransportBlock<dim, qdim>;
    FixedSourceProblem<dim, qdim, TransportType, TransportBlockType> problem(
        dof_handler, quadrature, mgxs_pseudo, boundary_conditions_one);
    pgd::sn::FixedSourceP fixed_source_p(
        problem.fixed_source, mgxs_pseudo, mgxs_one, sources_spaceangle);
    pgd::sn::EnergyMgFull energy_mg(*mgxs, sources_energy);
    std::vector<pgd::sn::LinearInterface*> linear_ops = 
        {&energy_mg, &fixed_source_p};
    pgd::sn::NonlinearGS nonlinear_gs(linear_ops, num_materials, 1, num_sources);
    // pgd main loop
    dealii::Vector<double> spectrum_one;
    std::vector<dealii::BlockVector<double>> modes_spaceangle;
    std::vector<Mgxs> mgxs_coarses;
    std::vector<Mgxs> mgxs_coarses_ip;
    dealii::BlockVector<double> _;
    for (int m = 0; m < num_modes; ++m) {
      nonlinear_gs.enrich();
      std::cout << "mode " << m << std::endl;
      for (int k = 0; k < max_iters_nonlinear; ++k) {
        double residual = std::numeric_limits<double>::infinity();
        try {
          residual = nonlinear_gs.step(_, _);
        } catch (dealii::SolverControl::NoConvergence &failure) {
          failure.print_info(std::cout);
          break;
        }
        std::cout << "picard " << k << ": " << residual << std::endl;
        if (residual < tol_nonlinear)
          break;
      }
      nonlinear_gs.finalize();
      if (do_update) {
        nonlinear_gs.update();
      }
      if (m == 0)
        spectrum_one = energy_mg.modes.front();
      // post-process
      modes_spaceangle.emplace_back(quadrature.size(), dof_handler.n_dofs());
      modes_spaceangle.back() = fixed_source_p.caches.back().mode.block(0);
      GetMgxsCoarses(mgxs_coarses, modes_spaceangle, energy_mg.modes, 
                     problem_full.transport, problem.d2m, *mgxs, g_maxes, m);
      GetMgxsCoarses(mgxs_coarses_ip, modes_spaceangle, energy_mg.modes,
                     problem_full.transport, problem.d2m, *mgxs, g_maxes, m,
                     1, INCONSISTENT_P);
    }
    std::cout << "done running pgd\n";
    // Post-process spectrum
    dealii::ConvergenceTable table_spectrum;
    for (int j = 0; j < num_materials; ++j) {
      std::string key = "j" + std::to_string(j);
      for (int g = 0; g < num_groups; ++g) {
        table_spectrum.add_value(key, spectra[j].block(0)[g]);
      }
      table_spectrum.set_scientific(key, true);
      table_spectrum.set_precision(key, 16);
    }
    for (int g = 0; g < num_groups; ++g) {
      table_spectrum.add_value("inf", spectrum[g]);
      table_spectrum.add_value("mode1u", energy_mg.modes.front()[g]);
      table_spectrum.add_value("mode1p", spectrum_one[g]);
    }
    for (const std::string key : {"inf", "mode1u", "mode1p"}) {
      table_spectrum.set_scientific(key, true);
      table_spectrum.set_precision(key, 16);
    }
    this->WriteConvergenceTable(table_spectrum, "_spectrum");
    // Post-process
    dealii::ConvergenceTable table;
    std::vector<double> l2_errors_d(num_modes+1);
    std::vector<double> l2_errors_m(num_modes+1);
    std::cout << "get l2 errors discrete\n";
    this->GetL2ErrorsDiscrete(l2_errors_d, modes_spaceangle, energy_mg.modes, 
                              flux_full, problem.transport, table, "error_d");
    this->GetL2ErrorsMoments(l2_errors_m, modes_spaceangle, energy_mg.modes, 
                             flux_full, problem.transport, problem.d2m, table, 
                             "error_m");
    std::cout << "getting mgxs coarses\n";
    std::vector<Mgxs> mgxs_coarses_svd;
    GetMgxsCoarses(mgxs_coarses_svd, svecs_spaceangle, svecs_energy,
                   problem_full.transport, problem.d2m, *mgxs, g_maxes);
    std::cout << "got mgxs coarses\n";
    for (int j = 0; j < num_materials; ++j) {
      if (j != 2)
        continue;
      for (int g_coarse = 0; g_coarse < g_maxes.size(); ++g_coarse) {
        if (g_coarse+1 < 21 || g_coarse+1 > 27)
          continue;
        std::string key = "j" + std::to_string(j) 
                        + "g" + std::to_string(g_coarse+1);
        table.add_value(key, std::nan("a"));
        table.add_value(key+"inf", std::nan("b"));
        // table.add_value(key+"svd", std::nan("c"));
        table.add_value(key+"ip", std::nan("d"));
        for (int m = 0; m < num_modes; ++m) {
          double full = mgxs_coarse.total[g_coarse][j];
          double error = (full - mgxs_coarses[m].total[g_coarse][j]) / full;
          table.add_value(key, error);
          double error_inf = (full - mgxs_coarse_inf.total[g_coarse][j]) / full;
          table.add_value(key+"inf", error_inf);
          // double error_svd = 
          //     (full - mgxs_coarses_svd[m].total[g_coarse][j]) / full;
          // table.add_value(key+"svd", error_svd);
          double full_ip = mgxs_coarse_ip.total[g_coarse][j];
          double error_ip = (full_ip - mgxs_coarses_ip[m].total[g_coarse][j]) 
                            / full_ip;
          table.add_value(key+"ip", error_ip);
        }
        table.set_scientific(key, true);
        table.set_precision(key, 16);
        table.set_scientific(key+"inf", true);
        table.set_precision(key+"inf", 16);
        // table.set_scientific(key+"svd", true);
        // table.set_precision(key+"svd", 16);
        table.set_scientific(key+"ip", true);
        table.set_precision(key+"ip", 16);
      }
    }
    this->WriteConvergenceTable(table);
  }

  void RunInfiniteMedium(dealii::Vector<double> &spectrum, 
                         const dealii::Vector<double> &source,
                         const Mgxs &mgxs, const std::vector<double> &volumes) {
    const int num_groups = mgxs.total.size();
    const int num_materials = mgxs.total[0].size();
    AssertDimension(spectrum.size(), num_groups);
    AssertDimension(volumes.size(), num_materials);
    // Assemble linear system
    dealii::FullMatrix<double> matrix(num_groups);
    for (int j = 0; j < num_materials; ++j) {
      for (int g = 0; g < num_groups; ++g) {
        matrix[g][g] += mgxs.total[g][j] * volumes[j];
        for (int gp = 0; gp < num_groups; ++gp) {
          matrix[g][gp] -= mgxs.scatter[g][gp][j] * volumes[j];
        }
      }
    }
    // Solve directly
    matrix.gauss_jordan();
    matrix.vmult(spectrum, source);
  }

  void GetMgxsCoarses(
      std::vector<Mgxs> &mgxs_coarses,
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const sn::Transport<dim, qdim> &transport,
      const sn::DiscreteToMoment<qdim> &d2m,
      const Mgxs &mgxs_fine,
      const std::vector<int> &g_maxes,
      const int m_start = 0,
      const int order = 0,
      const TransportCorrection correction = CONSISTENT_P) {
    const int num_modes = modes_spaceangle.size();
    const int num_groups = mgxs_fine.total.size();
    const int num_moments = d2m.n_block_rows(order);
    dealii::BlockVector<double> flux_l(num_groups, 
                                       (order+1) * dof_handler.n_dofs());
    dealii::BlockVector<double> flux_m(num_groups,
                                       num_moments * dof_handler.n_dofs());
    dealii::Vector<double> mode_m(num_moments * dof_handler.n_dofs());
    dealii::BlockVector<double> mode_mb(num_moments, dof_handler.n_dofs());
    for (int m = 0; m < num_modes; ++m) {
      d2m.vmult(mode_mb, modes_spaceangle[m]);
      mode_m = mode_mb;
      for (int g = 0; g < num_groups; ++g)
        flux_m.block(g).add(modes_energy[m][g], mode_m);
      if (m >= m_start) {
        for (int g = 0; g < num_groups; ++g)
          d2m.moment_to_legendre(flux_l.block(g), flux_m.block(g), order);
        Mgxs mgxs_coarse = collapse_mgxs(
            flux_l, dof_handler, transport, mgxs_fine, g_maxes, correction);
        mgxs_coarses.push_back(mgxs_coarse);
      }
    }
  }
};

#endif  // AETHER_EXAMPLES_MGXS_TEST_H_