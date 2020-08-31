#ifndef AETHER_EXAMPLES_COARSE_TEST_H_
#define AETHER_EXAMPLES_COARSE_TEST_H_

#include "compare_test.h"

template <int dim, int qdim>
class CoarseTest : virtual public CompareTest<dim, qdim> {
 protected:
  dealii::ConvergenceTable table;
  dealii::BlockVector<double> source_coarse;
  dealii::BlockVector<double> flux_coarsened;
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
                     const bool precomputed=false) {
    const int num_groups = mgxs->total.size();
    const int num_materials = mgxs->total[0].size();
    // for (int g = 0; g < num_groups; ++g) {
    //   for (int gp = 0; gp < num_groups; ++gp)
    //     std::cout << mgxs->scatter[g][gp][1] << " ";
    //   std::cout << "\n";
    // }
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
    dealii::BlockVector<double> flux_full(
        num_groups, quadrature.size()*dof_handler.n_dofs());
    dealii::BlockVector<double> source_full(flux_full.get_block_indices());
    for (int g = 0; g < num_groups; ++g)
      for (int j = 0; j < num_sources; ++j)
        source_full.block(g).add(
            sources_energy[j][g], sources_spaceangle[j].block(0));
    // using TransportType = pgd::sn::Transport<dim, qdim>;
    FixedSourceProblem<dim, qdim> problem_full(
        dof_handler, quadrature, *mgxs, boundary_conditions);
    // TransportType transport = problem_full.transport.transport;
    const std::string filename_h5 = this->GetTestName() + ".h5";
    namespace HDF5 = dealii::HDF5;
    if (precomputed) {
      HDF5::File file(filename_h5, HDF5::File::FileAccessMode::open);
      flux_full = file.open_dataset("flux_full").read<dealii::Vector<double>>();
    } else {
      std::vector<double> history_data;
      CompareTest<dim, qdim>::RunFullOrder(flux_full, source_full, problem_full, 
                                           max_iters_fullorder, tol_fullorder, 
                                           &history_data);
      this->WriteFlux(flux_full, history_data, filename_h5);
    }
    this->PlotFlux(flux_full, problem_full.d2m, mgxs->group_structure, "full");
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
    write_mgxs(mgxs_coarse, filename, "294K", materials);
    Mgxs mgxs_coarse_ip = collapse_mgxs(
        flux_full_l1, dof_handler, problem_full.transport, *mgxs, g_maxes,
        INCONSISTENT_P);
    write_mgxs(mgxs_coarse_ip, test_name+"_ip_mgxs.h5", "294K", materials);
    std::cout << "MGXS TO FILE " << filename << std::endl;
    // Compute svd of full order
    std::vector<dealii::BlockVector<double>> svecs_spaceangle;
    std::vector<dealii::Vector<double>> svecs_energy;
    this->ComputeSvd(svecs_spaceangle, svecs_energy, flux_full, 
                     problem_full.transport);
    const int num_svecs = svecs_spaceangle.size();
    // Get coarsened quantities
    const int num_groups_coarse = g_maxes.size();
    std::vector<double> group_structure_coarse(num_groups_coarse+1);
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
    // run pgd
    // this->RunPgd(nonlinear_gs, num_modes, max_iters_nonlinear, tol_nonlinear,
    //              do_update);
    std::vector<int> m_coarses = {1, 10, 20, 30};
    const int incr = 10;
    std::vector<dealii::BlockVector<double>> modes_spaceangle;
    std::vector<Mgxs> mgxs_coarses;
    std::vector<Mgxs> mgxs_coarses_ip;
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
      if (do_update) {
        nonlinear_gs.finalize();
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
    }
    modes_spaceangle.clear();
    std::cout << "done running pgd\n";
    std::vector<dealii::BlockVector<double>> fluxes_coarsened_pgd;
    std::vector<dealii::BlockVector<double>> fluxes_coarsened_svd;
    dealii::BlockVector<double> flux_coarsened_pgd(
        flux_coarsened.get_block_indices());
    dealii::BlockVector<double> flux_coarsened_svd(flux_coarsened_pgd);
    for (int m = 0; m < num_modes; ++m) {
      dealii::Vector<double> svec_spaceangle(svecs_spaceangle[m].size());
      svec_spaceangle = svecs_spaceangle[m];
      for (int g_coarse = 0; g_coarse < num_groups_coarse; ++g_coarse) {
        int g_min = g_coarse == 0 ? 0 : g_maxes[g_coarse-1];
        int g_max = g_maxes[g_coarse];
        for (int g = g_min; g < g_max; ++g) {
          flux_coarsened_pgd.block(g_coarse).add(
              energy_mg.modes[m][g], fixed_source_p.caches[m].mode.block(0));
          flux_coarsened_svd.block(g_coarse).add(
              svecs_energy[m][g], svec_spaceangle);
        }
      }
      if ((m+1) % incr == 0) {
        fluxes_coarsened_pgd.push_back(flux_coarsened_pgd);
        fluxes_coarsened_svd.push_back(flux_coarsened_svd);
      }
    }
    // Run coarse group
    std::vector<std::vector<dealii::BlockVector<double>>>
        boundary_conditions_coarse(num_groups_coarse);
    FixedSourceProblem<dim, qdim> problem_coarse(
          dof_handler, quadrature, mgxs_coarse, boundary_conditions_coarse);
    dealii::BlockVector<double> flux_coarse_cp(
        flux_coarsened.get_block_indices());
    this->RunFullOrder(flux_coarse_cp, source_coarse, problem_coarse,
                       max_iters_fullorder, tol_fullorder);
    // Run coarse group, inconsistent P
    FixedSourceProblem<dim, qdim> problem_coarse_ip(
          dof_handler, quadrature, mgxs_coarse_ip, boundary_conditions_coarse);
    dealii::BlockVector<double> flux_coarse_ip(
        flux_coarse_cp.get_block_indices());
    this->RunFullOrder(flux_coarse_ip, source_coarse, problem_coarse_ip, 
                       max_iters_fullorder, tol_fullorder);
    // Run coarse group with PGD cross-sections
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
    // pgd::sn::Transport<dim, qdim> transport(dof_handler, quadrature);
    TransportType& transport = problem.transport;
    std::cout << "init'd transport\n";
    DiscreteToMoment<qdim> &d2m = problem_full.d2m;
    std::cout << "init'd d2m\n";
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
                                "coarse_d_abs"+label);
      GetL2ErrorsCoarseMoments(l2_errors_coarse_m_abs, flux_coarse, 
                              flux_coarsened, transport, d2m, false, table, 
                              "coarse_m_abs"+label);
      GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_rel, flux_coarse, 
                                flux_coarsened, transport, true, table, 
                                "coarse_d_rel"+label);
      GetL2ErrorsCoarseMoments(l2_errors_coarse_m_rel, flux_coarse, 
                               flux_coarsened, transport, d2m, true, table,
                               "coarse_m_rel"+label);
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
        // std::string m = "_m" + std::to_string((i + 1) * incr);
        std::string labelm = label+ "_m" + std::to_string(m_coarses[i]);
        GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_abs_mi, flux_coarses[i], 
                                  flux_coarsened, transport, false, table, 
                                  "coarse_d_abs"+labelm);
        GetL2ErrorsCoarseMoments(l2_errors_coarse_m_abs_mi, flux_coarses[i], 
                                 flux_coarsened, transport, d2m, false, table, 
                                 "coarse_m_abs"+labelm);
        GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_rel_mi, flux_coarses[i], 
                                  flux_coarsened, transport, true, table, 
                                  "coarse_d_rel"+labelm);
        GetL2ErrorsCoarseMoments(l2_errors_coarse_m_rel_mi, flux_coarses[i], 
                                 flux_coarsened, transport, d2m, true, table,
                                 "coarse_m_rel"+labelm);
      }
    }
    for (int d = 0; d < 2; ++d) {
      std::vector<dealii::BlockVector<double>> &fluxes_coarsened = 
          d ? fluxes_coarsened_svd : fluxes_coarsened_pgd;
      const std::string decomp = d ? "svd" : "pgd";                 
      for (int i = 0; i < fluxes_coarsened.size(); ++i) {
        std::vector<double> l2_errors_decomp_d_rel;
        std::vector<double> l2_errors_decomp_m_rel;
        std::vector<double> l2_errors_decomp_d_abs;
        std::vector<double> l2_errors_decomp_m_abs;
        std::string m = "_m" + std::to_string((i + 1) * incr);
        GetL2ErrorsCoarseDiscrete(l2_errors_decomp_d_abs, fluxes_coarsened[i], 
                                  flux_coarsened, transport, false, table,
                                  decomp+"_d_abs"+m);
        GetL2ErrorsCoarseMoments(l2_errors_decomp_m_abs, fluxes_coarsened[i], 
                                flux_coarsened, transport, d2m, false, table,
                                decomp+"_m_abs"+m);
        GetL2ErrorsCoarseDiscrete(l2_errors_decomp_d_rel, fluxes_coarsened[i], 
                                  flux_coarsened, transport, true, table, 
                                  decomp+"_d_rel"+m);
        GetL2ErrorsCoarseMoments(l2_errors_decomp_m_rel, fluxes_coarsened[i], 
                                flux_coarsened, transport, d2m, true, table, 
                                decomp+"_m_rel"+m);
        this->PlotFlux(fluxes_coarsened[i], problem_full.d2m, 
                       group_structure_coarse, decomp+"_coarsened"+m);
        this->PlotDiffAngular(fluxes_coarsened[i], flux_coarsened, 
            problem_full.d2m, decomp+"_diff_angular_coarsened"+m);
        this->PlotDiffScalar(fluxes_coarsened[i], flux_coarsened, 
            problem_full.d2m, decomp+"_diff_scalar_coarsened"+m);
      }
    }
    for (int g = 0; g < num_groups_coarse; ++g) {
      table.add_value("flux_coarse", flux_coarse_cp.block(g).l2_norm());
      table.add_value("flux_coarsened", flux_coarsened.block(g).l2_norm());
      // table.add_value("source_coarse", source_coarse.block(g).l2_norm());
    }
    table.set_scientific("flux_coarse", true);
    table.set_scientific("flux_coarsened", true);
    table.set_scientific("source_coarse", true);
    table.set_precision("flux_coarse", 16);
    table.set_precision("flux_coarsened", 16);
    // table.set_precision("source_coarse", 16);
    this->WriteConvergenceTable(table);
  }

  void GetL2ErrorsCoarseDiscrete(
      std::vector<double> &l2_errors,
      const dealii::BlockVector<double> &flux_coarse,
      const dealii::BlockVector<double> &flux_coarsened,
      const pgd::sn::Transport<dim, qdim> &transport,
      const bool is_relative,
      dealii::ConvergenceTable &table,
      const std::string &key) {
    const int num_groups = flux_coarse.n_blocks();
    AssertDimension(num_groups, flux_coarsened.n_blocks());
    l2_errors.resize(num_groups);
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
      if (is_relative)
        l2_errors[g] /= std::sqrt(l2_norm);
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
      const std::string &key) {
    const int num_groups = flux_coarse.n_blocks();
    AssertDimension(num_groups, flux_coarsened.n_blocks());
    l2_errors.resize(num_groups);
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
      table.add_value(key, l2_errors[g]);
    }
    table.set_scientific(key, true);
    table.set_precision(key, 16);
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