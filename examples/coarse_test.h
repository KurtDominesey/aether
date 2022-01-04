#ifndef AETHER_EXAMPLES_COARSE_TEST_H_
#define AETHER_EXAMPLES_COARSE_TEST_H_

#include "compare_test.h"

template <int dim, int qdim>
class CoarseTest : virtual public CompareTest<dim, qdim> {
 protected:
  dealii::ConvergenceTable table;
  dealii::BlockVector<double> source_coarse;
  dealii::BlockVector<double> flux_coarsened;
  std::vector<double> group_structure_coarse;
  using CompareTest<dim, qdim>::mesh;
  using CompareTest<dim, qdim>::quadrature;
  using CompareTest<dim, qdim>::dof_handler;
  using CompareTest<dim, qdim>::mgxs;

  void CompareCoarse(const int num_modes,
                     const int max_iters_nonlinear,
                     const double tol_nonlinear,
                     const bool do_update,
                     const int max_iters_fullorder,
                     const double tol_fullorder,
                     const std::vector<int> &g_maxes,
                     const std::vector<std::string> &materials,
                     std::vector<double> factors = {},
                     const bool precomputed_full=false,
                     const bool precomputed_cp=false,
                     const bool precomputed_ip=false,
                     const bool should_write_mgxs=true,
                     const bool do_eigenvalue=false) {
    const int num_groups = mgxs->total.size();
    const int num_materials = mgxs->total[0].size();
    // Create sources
    std::vector<dealii::Vector<double>> sources_energy;
    std::vector<dealii::BlockVector<double>> sources_spaceangle;
    this->WriteUniformFissionSource(sources_energy, sources_spaceangle);
    const int num_sources = sources_energy.size();
    if (factors.empty())
      factors.resize(num_sources, 1.0);
    AssertThrow(factors.size() == num_sources, dealii::ExcInvalidState());
    for (int j = 0; j < num_sources; ++j)
      sources_energy[j] *= factors[j];
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
    using TransportType = pgd::sn::Transport<dim, qdim>;
    FissionProblem<dim, qdim, TransportType> problem_full(
        dof_handler, quadrature, *mgxs, boundary_conditions);
    // TransportType transport = problem_full.transport.transport;
    double eigenvalue = 0;
    const std::string filename_h5 = this->GetTestName() + ".h5";
    namespace HDF5 = dealii::HDF5;
    if (precomputed_full) {
      HDF5::File file(filename_h5, HDF5::File::FileAccessMode::open);
      flux_full = file.open_dataset("flux_full").read<dealii::Vector<double>>();
      if (do_eigenvalue)
        eigenvalue = file.get_attribute<double>("k_eigenvalue");
    } else {
      std::vector<double> history_data;
      if (!do_eigenvalue)
        CompareTest<dim, qdim>::RunFullOrder(flux_full, source_full, problem_full, 
                                             max_iters_fullorder, tol_fullorder, 
                                             &history_data);
      else
        eigenvalue = CompareTest<dim, qdim>::RunFullOrderCriticality(
            flux_full, source_full, problem_full, max_iters_fullorder, 
            tol_fullorder, &history_data);
      this->WriteFlux(flux_full, history_data, filename_h5, eigenvalue);
    }
    this->PlotFlux(flux_full, problem_full.d2m, mgxs->group_structure, "full");
    // get k1 (first iteration eigenvalue)
    double denominator = 0;
    double k1 = this->ComputeEigenvalue(problem_full, flux_full, source_full, 
                                        *mgxs, denominator);
    // collapse mgxs
    dealii::BlockVector<double> flux_full_l0(num_groups, dof_handler.n_dofs());
    dealii::BlockVector<double> flux_full_l1(num_groups, 2*dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      problem_full.d2m.vmult(flux_full_l0.block(g), flux_full.block(g));
      problem_full.d2m.discrete_to_legendre(flux_full_l1.block(g), 
                                            flux_full.block(g));
    }
    std::cout << "collapsing spectra\n";
    std::vector<dealii::BlockVector<double>> spectra;
    collapse_spectra(spectra, flux_full_l0, dof_handler, 
                     problem_full.transport);
    Mgxs mgxs_coarse = collapse_mgxs(spectra, *mgxs, g_maxes);
    const std::string test_name = this->GetTestName();
    const std::string filename = test_name + "_mgxs.h5";
    Mgxs mgxs_coarse_ip = collapse_mgxs(
        flux_full_l1, dof_handler, problem_full.transport, *mgxs, g_maxes,
        INCONSISTENT_P);
    if (should_write_mgxs) {
      write_mgxs(mgxs_coarse, filename, "294K", materials);
      write_mgxs(mgxs_coarse_ip, test_name+"_ip_mgxs.h5", "294K", materials);
      std::cout << "MGXS TO FILE " << filename << std::endl;
    }
    // Print spectra
    dealii::ConvergenceTable table_spectra;
    for (int j = 0; j < num_materials; ++j) {
      std::string key = "j" + std::to_string(j);
      for (int g = 0; g < num_groups; ++g) {
        table_spectra.add_value(key, spectra[j].block(0)[g]);
      }
      table_spectra.set_scientific(key, true);
      table_spectra.set_precision(key, 16);
    }
    this->WriteConvergenceTable(table_spectra, "_spectra");
    // Compute svd of full order
    std::vector<dealii::BlockVector<double>> svecs_spaceangle;
    std::vector<dealii::Vector<double>> svecs_energy;
    this->ComputeSvd(svecs_spaceangle, svecs_energy, flux_full, 
                     problem_full.transport);
    const int num_svecs = svecs_spaceangle.size();
    // Get coarsened quantities
    const int num_groups_coarse = g_maxes.size();
    group_structure_coarse.resize(num_groups_coarse+1);
    group_structure_coarse[0] = mgxs->group_structure[0];
    flux_coarsened.reinit(
        num_groups_coarse, quadrature.size()*dof_handler.n_dofs());
    source_coarse.reinit(flux_coarsened.get_block_indices());
    for (int g_coarse = 0; g_coarse < num_groups_coarse; ++g_coarse) {
      int g_min = g_coarse == 0 ? 0 : g_maxes[g_coarse-1];
      int g_max = g_maxes[g_coarse];
      for (int g = g_min; g < g_max; ++g) {
        source_coarse.block(g_coarse) += source_full.block(g);
        flux_coarsened.block(g_coarse) += flux_full.block(g);
      }
      group_structure_coarse[g_coarse+1] = mgxs->group_structure[g_max];
    }
    dealii::Vector<double> produced;
    if (do_eigenvalue) {
      double norm = flux_coarsened.l2_norm();
      flux_coarsened /= norm;
      flux_full /= norm;
      dealii::BlockVector<double> flux_full_m(num_groups, dof_handler.n_dofs());
      for (int g = 0; g < num_groups; ++g)
        problem_full.d2m.vmult(flux_full_m.block(g), flux_full.block(g));
      produced.reinit(dof_handler.n_dofs());
      problem_full.production.vmult(produced, flux_full_m);
    }
    this->PlotFlux(flux_coarsened, problem_full.d2m, group_structure_coarse, 
                   "full_coarsened");
    // ERASE FULL-ORDER FINE-GROUP FLUX TO SAVE MEMORY
    flux_full.reinit(0);
    flux_full_l0.reinit(0);
    flux_full_l1.reinit(0);
    // Run pgd model
    Mgxs mgxs_one(1, num_materials, 1);
    Mgxs mgxs_pseudo(1, num_materials, 1);
    for (int j = 0; j < num_materials; ++j) {
      mgxs_one.total[0][j] = 1;
      mgxs_one.scatter[0][0][j] = 1;
      mgxs_one.chi[0][j] = 1;
      mgxs_one.nu_fission[0][j] = 1;
      mgxs_pseudo.chi[0][j] = 1;
    }
    using TransportType = pgd::sn::Transport<dim, qdim>;
    using TransportBlockType = pgd::sn::TransportBlock<dim, qdim>;
    FissionProblem<dim, qdim, TransportType, TransportBlockType> problem(
        dof_handler, quadrature, mgxs_pseudo, boundary_conditions_one);
    pgd::sn::FixedSourceP fixed_source_p(
        problem.fixed_source, mgxs_pseudo, mgxs_one, sources_spaceangle);
    pgd::sn::FissionSourceP fission_source_p(
        problem.fixed_source, problem.fission, mgxs_pseudo, mgxs_one);
    pgd::sn::EnergyMgFull energy_mg(*mgxs, sources_energy);
    pgd::sn::EnergyMgFiss energy_fiss(*mgxs);
    pgd::sn::EnergyMgFull& energy_op = 
        do_eigenvalue ? energy_fiss : energy_mg;
    pgd::sn::FixedSourceP<dim>& spatioangular_op = 
        do_eigenvalue ? fission_source_p : fixed_source_p;
    std::vector<pgd::sn::LinearInterface*> linear_ops = 
        {&energy_op, &spatioangular_op};
    pgd::sn::NonlinearGS fixed_gs(linear_ops, num_materials, 1, num_sources);
    pgd::sn::EigenGS eigen_gs(linear_ops, num_materials, 1);
    pgd::sn::NonlinearGS& nonlinear_gs = do_eigenvalue ? eigen_gs : fixed_gs;
    if (do_eigenvalue) {
      double k0 = eigen_gs.initialize_guess();
      std::cout << "initial k (guess): " << k0 << "\n";
    }
    // run pgd
    std::vector<double> eigenvalues;
    std::vector<int> m_coarses = {1, 10, 20, 30};
    const int incr = 10;
    std::vector<Mgxs> mgxs_coarses;
    std::vector<Mgxs> mgxs_coarses_ip;
    std::vector<dealii::BlockVector<double>> modes_spaceangle;
    if (do_eigenvalue) {
      std::vector<int> unconverged;
      std::vector<double> residuals;
      this->RunPgd(nonlinear_gs, num_modes, max_iters_nonlinear, tol_nonlinear,
                   do_update, unconverged, residuals, &eigenvalues);
      for (int m = 0; m < num_modes; ++m) {
        modes_spaceangle.emplace_back(quadrature.size(), dof_handler.n_dofs());
        modes_spaceangle.back() = spatioangular_op.caches[m].mode.block(0);
      }
    } else {
      // Can't call RunPgd because we need the modes before they're updated in
      // a later enrichment iteration.
      dealii::BlockVector<double> _;
      for (int m = 0; m < num_modes; ++m) {
        nonlinear_gs.enrich();
        for (int k = 0; k < max_iters_nonlinear; ++k) {
          try {
            double residual = nonlinear_gs.step(_, _);
            if (residual < tol_nonlinear)
              break;
          } catch (dealii::SolverControl::NoConvergence &failure) {
            failure.print_info(std::cout);
            break;
          }
        }
        nonlinear_gs.finalize();
        if (do_update) {
          nonlinear_gs.update();
        }
        // post-process
        modes_spaceangle.emplace_back(quadrature.size(), dof_handler.n_dofs());
        modes_spaceangle.back() = fixed_source_p.caches.back().mode.block(0);
        if (m+1 == m_coarses[mgxs_coarses.size()]) {
          mgxs_coarses.push_back(
              GetMgxsCoarse(modes_spaceangle, energy_mg.modes, 
                            problem_full.transport, problem.d2m, *mgxs, g_maxes, 
                            0, CONSISTENT_P));
          mgxs_coarses_ip.push_back(
              GetMgxsCoarse(modes_spaceangle, energy_mg.modes,
                            problem_full.transport, problem.d2m, *mgxs, g_maxes,
                            1, INCONSISTENT_P));
        }
        // get k1
        dealii::BlockVector<double> flux_pgd(
            num_groups, quadrature.size()*dof_handler.n_dofs());
        for (int mm = 0; mm <= m; ++mm) {
          for (int g = 0; g < num_groups; ++g)
            flux_pgd.block(g).add(energy_mg.modes[mm][g],
                                  fixed_source_p.caches[mm].mode.block(0));
        }
        double _ = 0;
        double k1_pgd = this->ComputeEigenvalue(problem_full, flux_pgd, 
                                                source_full, *mgxs, _);
      }
    }
    std::cout << "done running pgd\n";
    std::vector<dealii::BlockVector<double>> fluxes_coarsened_pgd;
    std::vector<dealii::BlockVector<double>> fluxes_coarsened_svd;
    dealii::BlockVector<double> flux_coarsened_pgd(
        flux_coarsened.get_block_indices());
    dealii::BlockVector<double> flux_coarsened_svd(flux_coarsened_pgd);
    std::vector<dealii::Vector<double>> modes_coarsened(
        num_modes, dealii::Vector<double>(num_groups_coarse));
    auto svecs_coarsened = modes_coarsened;
    for (int m = 0; m < num_modes; ++m) {
      dealii::Vector<double> svec_spaceangle(svecs_spaceangle[m].size());
      svec_spaceangle = svecs_spaceangle[m];
      for (int g_coarse = 0; g_coarse < num_groups_coarse; ++g_coarse) {
        int g_min = g_coarse == 0 ? 0 : g_maxes[g_coarse-1];
        int g_max = g_maxes[g_coarse];
        for (int g = g_min; g < g_max; ++g) {
          flux_coarsened_pgd.block(g_coarse).add(
              energy_op.modes[m][g], spatioangular_op.caches[m].mode.block(0));
          flux_coarsened_svd.block(g_coarse).add(
              svecs_energy[m][g], svec_spaceangle);
          if (do_eigenvalue) {
            modes_coarsened[m][g_coarse] += energy_op.modes[m][g];
            svecs_coarsened[m][g_coarse] += svecs_energy[m][g];
          }
        }
      }
      if ((m+1) % incr == 0 && !do_eigenvalue) {
        fluxes_coarsened_pgd.push_back(flux_coarsened_pgd);
        fluxes_coarsened_svd.push_back(flux_coarsened_svd);
      }
    }
    if (do_eigenvalue) {
      double sum = 0;
      for (const double &v: flux_coarsened_pgd)
        sum += v;
      double norm_pgd = flux_coarsened_pgd.l2_norm();
      double norm_svd = flux_coarsened_svd.l2_norm();
      if (sum < 0)
        norm_pgd *= -1;
      flux_coarsened_pgd /= norm_pgd;
      flux_coarsened_svd /= norm_svd;
      for (int m = 0; m < num_modes; ++m) {
        modes_coarsened[m] /= norm_pgd;
        svecs_coarsened[m] /= norm_svd;
      }
    }
    // Run coarse group
    std::vector<std::vector<dealii::BlockVector<double>>>
        boundary_conditions_coarse(num_groups_coarse);
    FissionProblem<dim, qdim> problem_coarse(
          dof_handler, quadrature, mgxs_coarse, boundary_conditions_coarse);
    dealii::BlockVector<double> flux_coarse_cp(
        flux_coarsened.get_block_indices());
    double eigenvalue_cp = 0;
    const std::string filename_cp_h5 = this->GetTestName() + "-cp.h5";
    if (precomputed_cp) {
      HDF5::File file(filename_cp_h5, HDF5::File::FileAccessMode::open);
      flux_coarse_cp = file.open_dataset("flux_full").read<dealii::Vector<double>>();
      if (do_eigenvalue)
        eigenvalue_cp = file.get_attribute<double>("k_eigenvalue");
    } else {
      std::vector<double> history_data;
      if (!do_eigenvalue)
        this->RunFullOrder(flux_coarse_cp, source_coarse, problem_coarse,
                          max_iters_fullorder, tol_fullorder, &history_data);
      else
        eigenvalue_cp = this->RunFullOrderCriticality(
            flux_coarse_cp, source_coarse, problem_coarse, max_iters_fullorder, 
            tol_fullorder, &history_data);
      this->WriteFlux(flux_coarse_cp, history_data, filename_cp_h5, eigenvalue_cp);
    }
    // Run coarse group, inconsistent P
    FissionProblem<dim, qdim> problem_coarse_ip(
          dof_handler, quadrature, mgxs_coarse_ip, boundary_conditions_coarse);
    dealii::BlockVector<double> flux_coarse_ip(
        flux_coarse_cp.get_block_indices());
    double eigenvalue_ip = 0;
    const std::string filename_ip_h5 = this->GetTestName() + "-ip.h5";
    if (precomputed_ip) {
      HDF5::File file(filename_ip_h5, HDF5::File::FileAccessMode::open);
      flux_coarse_ip = file.open_dataset("flux_full").read<dealii::Vector<double>>();
      if (do_eigenvalue)
        eigenvalue_ip = file.get_attribute<double>("k_eigenvalue");
    } else {
      std::vector<double> history_data;
      if (!do_eigenvalue)
        this->RunFullOrder(flux_coarse_ip, source_coarse, problem_coarse_ip, 
                          max_iters_fullorder, tol_fullorder, &history_data);
      else
        eigenvalue_ip = this->RunFullOrderCriticality(
            flux_coarse_ip, source_coarse, problem_coarse_ip, 
            max_iters_fullorder, tol_fullorder, &history_data);
      this->WriteFlux(flux_coarse_ip, history_data, filename_ip_h5, eigenvalue_ip);
    }
    // Run coarse group with PGD cross-sections
    if (do_eigenvalue)
      mgxs_coarses.clear();
    std::vector<dealii::BlockVector<double>> flux_coarses_cp(mgxs_coarses.size(),
        flux_coarse_cp.get_block_indices());
    std::vector<dealii::BlockVector<double>> flux_coarses_ip(flux_coarses_cp);
    for (int i = 0; i < mgxs_coarses.size(); ++i) {
      FixedSourceProblem<dim, qdim> problem_coarse_m(
          dof_handler, quadrature, mgxs_coarses[i], boundary_conditions_coarse);
      this->RunFullOrder(flux_coarses_cp[i], source_coarse, problem_coarse_m,
                         max_iters_fullorder, tol_fullorder);
      FixedSourceProblem<dim, qdim> problem_coarse_ip_m(
          dof_handler, quadrature, mgxs_coarses_ip[i], 
          boundary_conditions_coarse);
      this->RunFullOrder(flux_coarses_ip[i], source_coarse, problem_coarse_ip_m,
                         max_iters_fullorder, tol_fullorder);
    }
    // Post process
    std::cout << "post-processing\n";
    std::vector<double> lethargy_widths(num_groups_coarse);
    for (int g = 0; g < num_groups_coarse; ++g) {
      int g_rev = num_groups - 1 - g;
      lethargy_widths[g] = std::log(
          mgxs->group_structure[g_rev+1]/mgxs->group_structure[g_rev]);
    }
    std::vector<double> l2_norms_d;
    std::vector<double> l2_norms_m;
    TransportType& transport = problem.transport;
    std::cout << "init'd transport\n";
    DiscreteToMoment<qdim> &d2m = problem_full.d2m;
    std::cout << "init'd d2m\n";
    std::vector<double> eigenvalues_coarse = {eigenvalue_cp, eigenvalue_ip};
    std::vector<std::string> labels = {"cp", "ip"};
    for (int c = 0; c < labels.size(); ++c) {
      dealii::BlockVector<double> &flux_coarse = 
          !c ? flux_coarse_cp : flux_coarse_ip;
      std::vector<dealii::BlockVector<double>> &flux_coarses =
          !c ? flux_coarses_cp : flux_coarses_ip;
      std::string label = "_" + labels[c];
      std::vector<double> l2_errors_coarse_d_rel;
      std::vector<double> l2_errors_coarse_m_rel;
      std::vector<double> l2_errors_coarse_d_abs;
      std::vector<double> l2_errors_coarse_m_abs;
      GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_abs, flux_coarse, 
                                flux_coarsened, transport, false, table, 
                                "coarse_d_abs"+label, l2_norms_d);
      GetL2ErrorsCoarseMoments(l2_errors_coarse_m_abs, flux_coarse, 
                              flux_coarsened, transport, d2m, false, table, 
                              "coarse_m_abs"+label, l2_norms_m);
      if (do_eigenvalue) {
        double l2_error_d = 0;
        double l2_error_m = 0;
        for (int g = 0; g < num_groups_coarse; ++g) {
          l2_error_d += std::pow(l2_errors_coarse_d_abs[g], 2) 
                        / lethargy_widths[g];
          l2_error_m += std::pow(l2_errors_coarse_m_abs[g], 2)
                        / lethargy_widths[g];
        }
        l2_error_d = std::sqrt(l2_error_d);
        l2_error_m = std::sqrt(l2_error_m);
        double l2_error_q = GetL2ErrorCoarseFissionSource(
            flux_coarse, produced, transport, d2m, problem_coarse.production);
        double dk = (eigenvalues_coarse[c] - eigenvalue) * 1e5;
        std::cout << labels[c] << "\n" << "l2 error (d, m, q): " 
                  << std::scientific
                  << l2_error_d << ", " 
                  << l2_error_m << ", " 
                  << l2_error_q << "\n"
                  << "dk [pcm]: " << dk << "\n";
      }
      GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_rel, flux_coarse, 
                                flux_coarsened, transport, true, table, 
                                "coarse_d_rel"+label, l2_norms_d);
      GetL2ErrorsCoarseMoments(l2_errors_coarse_m_rel, flux_coarse, 
                               flux_coarsened, transport, d2m, true, table,
                               "coarse_m_rel"+label, l2_norms_m);
      this->PlotFlux(flux_coarse, problem_full.d2m, 
                      group_structure_coarse, "coarse"+label);
      this->PlotDiffAngular(flux_coarse, flux_coarsened, problem_full.d2m,
                            "diff_angular_coarse"+label);
      this->PlotDiffScalar(flux_coarse, flux_coarsened, problem_full.d2m,
                           "diff_scalar_coarse"+label);
      for (int i = 0; i < mgxs_coarses.size(); ++i) {
        std::vector<double> l2_errors_coarse_d_rel_mi;
        std::vector<double> l2_errors_coarse_m_rel_mi;
        std::vector<double> l2_errors_coarse_d_abs_mi;
        std::vector<double> l2_errors_coarse_m_abs_mi;
        std::string labelm = label+ "_m" + std::to_string(m_coarses[i]);
        GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_abs_mi, flux_coarses[i], 
                                  flux_coarsened, transport, false, table, 
                                  "coarse_d_abs"+labelm, l2_norms_d);
        GetL2ErrorsCoarseMoments(l2_errors_coarse_m_abs_mi, flux_coarses[i], 
                                 flux_coarsened, transport, d2m, false, table, 
                                 "coarse_m_abs"+labelm, l2_norms_m);
        GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_rel_mi, flux_coarses[i], 
                                  flux_coarsened, transport, true, table, 
                                  "coarse_d_rel"+labelm, l2_norms_d);
        GetL2ErrorsCoarseMoments(l2_errors_coarse_m_rel_mi, flux_coarses[i], 
                                 flux_coarsened, transport, d2m, true, table,
                                 "coarse_m_rel"+labelm, l2_norms_m);
      }
    }
    dealii::ConvergenceTable table_modal;
    for (int d = 0; d < 2; ++d) {
      std::vector<dealii::BlockVector<double>> &fluxes_coarsened = 
          d ? fluxes_coarsened_svd : fluxes_coarsened_pgd;
      const std::string decomp = d ? "svd" : "pgd";
      if (do_eigenvalue) {
        if (decomp != "pgd")
          continue; // TODO: impl for svd
        std::vector<double> l2_errors_d(num_modes+1);
        auto l2_errors_m = l2_errors_d;
        auto l2_errors_q = l2_errors_d;
        this->GetL2ErrorsDiscrete(
            l2_errors_d, modes_spaceangle, modes_coarsened, flux_coarsened, 
            transport, table_modal, "l2_error_d");
        this->GetL2ErrorsMoments(
            l2_errors_m, modes_spaceangle, modes_coarsened, flux_coarsened,
            transport, d2m, table_modal, "l2_error_m");
        this->GetL2ErrorsFissionSource(
            l2_errors_q, modes_spaceangle, modes_coarsened, flux_coarsened,
            transport, d2m, problem_coarse.production, table_modal, 
            "l2_error_q");
        table_modal.add_value("error_k", std::nan("k"));
        for (int m = 0; m < num_modes; ++m)
          table_modal.add_value("error_k", 1e5*(eigenvalues[m]-eigenvalue));
        table_modal.set_scientific("error_k", true);
        table_modal.set_precision("error_k", 16);
        for (const std::string suffix: {"d", "m", "q"}) {
          const std::string key = "l2_error_" + suffix;
          table_modal.set_scientific(key, true);
          table_modal.set_precision(key, 16);
        }
      }              
      for (int i = 0; i < fluxes_coarsened.size(); ++i) {
        std::vector<double> l2_errors_decomp_d_rel;
        std::vector<double> l2_errors_decomp_m_rel;
        std::vector<double> l2_errors_decomp_d_abs;
        std::vector<double> l2_errors_decomp_m_abs;
        std::string m = "_m" + std::to_string((i + 1) * incr);
        GetL2ErrorsCoarseDiscrete(l2_errors_decomp_d_abs, fluxes_coarsened[i], 
                                  flux_coarsened, transport, false, table,
                                  decomp+"_d_abs"+m, l2_norms_d);
        GetL2ErrorsCoarseMoments(l2_errors_decomp_m_abs, fluxes_coarsened[i], 
                                flux_coarsened, transport, d2m, false, table,
                                decomp+"_m_abs"+m, l2_norms_m);
        GetL2ErrorsCoarseDiscrete(l2_errors_decomp_d_rel, fluxes_coarsened[i], 
                                  flux_coarsened, transport, true, table, 
                                  decomp+"_d_rel"+m, l2_norms_d);
        GetL2ErrorsCoarseMoments(l2_errors_decomp_m_rel, fluxes_coarsened[i], 
                                flux_coarsened, transport, d2m, true, table, 
                                decomp+"_m_rel"+m, l2_norms_m);
        this->PlotFlux(fluxes_coarsened[i], problem_full.d2m, 
                       group_structure_coarse, decomp+"_coarsened"+m);
        this->PlotDiffAngular(fluxes_coarsened[i], flux_coarsened, 
            problem_full.d2m, decomp+"_diff_angular_coarsened"+m);
        this->PlotDiffScalar(fluxes_coarsened[i], flux_coarsened, 
            problem_full.d2m, decomp+"_diff_scalar_coarsened"+m);
      }
    }
    // */
    for (int g = 0; g < num_groups_coarse; ++g) {
      table.add_value("flux_coarsened", flux_coarsened.block(g).l2_norm());
      table.add_value("source_coarse", source_coarse.block(g).l2_norm());
      table.add_value("l2_norm_d", l2_norms_d[g]);
      table.add_value("l2_norm_m", l2_norms_m[g]);
    }
    table.set_scientific("flux_coarsened", true);
    table.set_scientific("source_coarse", true);
    table.set_scientific("l2_norm_d", true);
    table.set_scientific("l2_norm_m", true);
    table.set_precision("flux_coarsened", 16);
    table.set_precision("source_coarse", 16);
    table.set_precision("l2_norm_d", 16);
    table.set_precision("l2_norm_m", 16);
    this->WriteConvergenceTable(table);
    if (do_eigenvalue)
      this->WriteConvergenceTable(table_modal, "-modal");
  }

  void GetL2ErrorsCoarseDiscrete(
      std::vector<double> &l2_errors,
      const dealii::BlockVector<double> &flux_coarse,
      const dealii::BlockVector<double> &flux_coarsened,
      const pgd::sn::Transport<dim, qdim> &transport,
      const bool is_relative,
      dealii::ConvergenceTable &table,
      const std::string &key,
      std::vector<double> &l2_norms) {
    const int num_groups = flux_coarse.n_blocks();
    AssertDimension(num_groups, flux_coarsened.n_blocks());
    l2_errors.resize(num_groups);
    l2_norms.clear();
    l2_norms.resize(num_groups);
    dealii::BlockVector<double> diff(quadrature.size(), dof_handler.n_dofs());
    dealii::BlockVector<double> flux_coarsened_g(diff);
    dealii::Vector<double> diff_l2(dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      double l2_norm = 0;
      diff = flux_coarse.block(g);
      flux_coarsened_g = flux_coarsened.block(g);
      diff -= flux_coarsened_g;
      for (int n = 0; n < quadrature.size(); ++n) {
        transport.collide_ordinate(diff_l2, diff.block(n));
        l2_errors[g] += quadrature.weight(n) * (diff.block(n) * diff_l2);
        transport.collide_ordinate(diff_l2, flux_coarsened_g.block(n));
        l2_norm += quadrature.weight(n) * (flux_coarsened_g.block(n) * diff_l2);
      }
      l2_errors[g] = std::sqrt(l2_errors[g]);
      l2_norms[g] = std::sqrt(l2_norm);
      if (is_relative)
        l2_errors[g] /= l2_norms[g];
      table.add_value(key, l2_errors[g]);
    }
    table.set_scientific(key, true);
    table.set_precision(key, 16);
  }

  void GetL2ErrorsCoarseMoments(
      std::vector<double> &l2_errors,
      const dealii::BlockVector<double> &flux_coarse,
      const dealii::BlockVector<double> &flux_coarsened,
      const pgd::sn::Transport<dim, qdim> &transport,
      const sn::DiscreteToMoment<qdim> &d2m,
      const bool is_relative,
      dealii::ConvergenceTable &table,
      const std::string &key,
      std::vector<double> &l2_norms) {
    const int num_groups = flux_coarse.n_blocks();
    AssertDimension(num_groups, flux_coarsened.n_blocks());
    l2_errors.resize(num_groups);
    l2_norms.clear();
    l2_norms.resize(num_groups);
    dealii::Vector<double> diff(dof_handler.n_dofs());
    dealii::Vector<double> diff_l2(diff);
    dealii::Vector<double> flux_coarsened_g(diff);
    for (int g = 0; g < num_groups; ++g) {
      d2m.vmult(diff, flux_coarse.block(g));
      d2m.vmult(flux_coarsened_g, flux_coarsened.block(g));
      diff -= flux_coarsened_g;
      transport.collide_ordinate(diff_l2, diff);
      double l2_error = std::sqrt(diff * diff_l2);
      transport.collide_ordinate(diff_l2, flux_coarsened_g);
      double l2_norm = std::sqrt(flux_coarsened_g * diff_l2);
      if (is_relative)
        l2_error /= l2_norm;
      l2_errors[g] = l2_error;
      l2_norms[g] = l2_norm;
      table.add_value(key, l2_errors[g]);
    }
    table.set_scientific(key, true);
    table.set_precision(key, 16);
  }

  double GetL2ErrorCoarseFissionSource(
      const dealii::BlockVector<double> &flux_coarse,
      const dealii::Vector<double> &produced_fine,
      const pgd::sn::Transport<dim, qdim> &transport,
      const sn::DiscreteToMoment<qdim> &d2m,
      const sn::Production<dim> &production) {
    const int num_groups = flux_coarse.n_blocks();
    dealii::BlockVector<double> flux_coarse_m(num_groups, dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g)
      d2m.vmult(flux_coarse_m.block(g), flux_coarse.block(g));
    dealii::Vector<double> diff(dof_handler.n_dofs());
    dealii::Vector<double> diff_l2(diff);
    production.vmult(diff, flux_coarse_m);
    diff -= produced_fine;
    transport.collide_ordinate(diff_l2, diff);
    return std::sqrt(diff * diff_l2);
  }

  Mgxs GetMgxsCoarse(
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const sn::Transport<dim, qdim> &transport,
      const sn::DiscreteToMoment<qdim> &d2m,
      const Mgxs &mgxs_fine,
      const std::vector<int> &g_maxes,
      const int order,
      const TransportCorrection correction) {
    const int num_modes = modes_spaceangle.size();
    const int num_groups = mgxs_fine.total.size();
    const int num_moments = d2m.n_block_rows(order);
    dealii::BlockVector<double> flux_l(
        num_groups, (order+1) * dof_handler.n_dofs());
    dealii::BlockVector<double> flux_m(
        num_groups, num_moments * dof_handler.n_dofs());
    dealii::BlockVector<double> mode_mb(num_moments, dof_handler.n_dofs());
    dealii::Vector<double> mode_m(num_moments * dof_handler.n_dofs());
    for (int m = 0; m < num_modes; ++m) {
      d2m.vmult(mode_mb, modes_spaceangle[m]);
      mode_m = mode_mb;
      for (int g = 0; g < num_groups; ++g)
        flux_m.block(g).add(modes_energy[m][g], mode_m);
    }
    for (int g = 0; g < num_groups; ++g)
      d2m.moment_to_legendre(flux_l.block(g), flux_m.block(g), order);
    return collapse_mgxs(flux_l, dof_handler, transport, mgxs_fine, g_maxes,
                         correction);
  }
};

#endif  // AETHER_EXAMPLES_COARSE_TEST_H_