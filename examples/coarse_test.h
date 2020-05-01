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
                     const std::vector<std::string> &materials) {
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
    CompareTest<dim, qdim>::RunFullOrder(flux_full, source_full, problem_full, 
                       max_iters_fullorder, tol_fullorder);
    dealii::BlockVector<double> flux_full_iso(num_groups, dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g)
      problem_full.d2m.vmult(flux_full_iso.block(g), flux_full.block(g));
    // Mgxs mgxs_coarse = collapse_mgxs(
    //       flux_full_iso, dof_handler, problem_full.transport, *mgxs, g_maxes);
    std::cout << "collapsing spectra\n";
    std::vector<dealii::Vector<double>> spectra;
    collapse_spectra(spectra, flux_full_iso, dof_handler, 
                     problem_full.transport);
    Mgxs mgxs_coarse = collapse_mgxs(spectra, *mgxs, g_maxes);
    const std::string test_name = this->GetTestName();
    const std::string filename = materials[2] + "_mgxs.h5";
    write_mgxs(mgxs_coarse, filename, "294K", materials);
    std::cout << "MGXS TO FILE " << filename << std::endl;
    // Run coarse group
    const int num_groups_coarse = g_maxes.size();
    dealii::BlockVector<double> flux_coarse(
        num_groups_coarse, quadrature.size() * dof_handler.n_dofs());
    source_coarse.reinit(flux_coarse.get_block_indices());
    flux_coarsened.reinit(flux_coarse.get_block_indices());
    for (int g_coarse = 0; g_coarse < num_groups_coarse; ++g_coarse) {
      int g_min = g_coarse == 0 ? 0 : g_maxes[g_coarse-1];
      int g_max = g_maxes[g_coarse];
      for (int g = g_min; g < g_max; ++g) {
        source_coarse.block(g_coarse) += source_full.block(g);
        flux_coarsened.block(g_coarse) += flux_full.block(g);
      }
    }
    std::vector<std::vector<dealii::BlockVector<double>>>
        boundary_conditions_coarse(num_groups_coarse);
    FixedSourceProblem<dim, qdim> problem_coarse(
          dof_handler, quadrature, mgxs_coarse, boundary_conditions_coarse);
    this->RunFullOrder(flux_coarse, source_coarse, problem_coarse,
                       max_iters_fullorder, tol_fullorder);
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
        {&fixed_source_p, &energy_mg};
    pgd::sn::NonlinearGS nonlinear_gs(linear_ops, num_materials, 1, num_sources);
    this->RunPgd(nonlinear_gs, num_modes, max_iters_nonlinear, tol_nonlinear,
                 do_update);
    std::cout << "done running pgd\n";
    dealii::BlockVector<double> flux_coarsened_pgd(
        flux_coarse.get_block_indices());
    for (int g_coarse = 0; g_coarse < num_groups_coarse; ++g_coarse) {
      int g_min = g_coarse == 0 ? 0 : g_maxes[g_coarse-1];
      int g_max = g_maxes[g_coarse];
      for (int g = g_min; g < g_max; ++g)
        for (int m = 0; m < num_modes; ++m)
          flux_coarsened_pgd.block(g_coarse).add(
              energy_mg.modes[m][g], fixed_source_p.caches[m].mode.block(0));
    }
    // Post process
    std::cout << "post-processing\n";
    pgd::sn::Transport<dim, qdim> transport(dof_handler, quadrature);
    DiscreteToMoment<qdim> &d2m = problem_full.d2m;
    std::vector<double> l2_errors_coarse_d_rel;
    std::vector<double> l2_errors_coarse_m_rel;
    std::vector<double> l2_errors_coarse_d_abs;
    std::vector<double> l2_errors_coarse_m_abs;
    std::vector<double> l2_errors_pgd_d_rel;
    std::vector<double> l2_errors_pgd_m_rel;
    std::vector<double> l2_errors_pgd_d_abs;
    std::vector<double> l2_errors_pgd_m_abs;
    GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_abs, flux_coarse, 
                              flux_coarsened, transport, false, table, 
                              "coarse_d_abs");
    GetL2ErrorsCoarseMoments(l2_errors_coarse_m_abs, flux_coarse, 
                             flux_coarsened, transport, d2m, false, table, 
                             "coarse_m_abs");
    GetL2ErrorsCoarseDiscrete(l2_errors_coarse_d_rel, flux_coarse, 
                              flux_coarsened, transport, true, table, 
                              "coarse_d_rel");
    GetL2ErrorsCoarseMoments(l2_errors_coarse_m_rel, flux_coarse, 
                             flux_coarsened, transport, d2m, true, table,
                             "coarse_m_rel");
    GetL2ErrorsCoarseDiscrete(l2_errors_pgd_d_abs, flux_coarsened_pgd, 
                              flux_coarsened, transport, false, table,
                              "pgd_d_abs");
    GetL2ErrorsCoarseMoments(l2_errors_pgd_m_abs, flux_coarsened_pgd, 
                             flux_coarsened, transport, d2m, false, table,
                             "pgd_m_abs");
    GetL2ErrorsCoarseDiscrete(l2_errors_pgd_d_rel, flux_coarsened_pgd, 
                              flux_coarsened, transport, true, table, 
                              "pgd_d_rel");
    GetL2ErrorsCoarseMoments(l2_errors_pgd_m_rel, flux_coarsened_pgd, 
                             flux_coarsened, transport, d2m, true, table, 
                             "pgd_m_rel");
    for (int g = 0; g < num_groups_coarse; ++g) {
      table.add_value("flux_coarse", flux_coarse.block(g).l2_norm());
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
        transport.collide(diff_l2, diff.block(n));
        l2_errors[g] += quadrature.weight(n) * (diff.block(n) * diff_l2);
        transport.collide(diff_l2, flux_coarsened_g.block(n));
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
      transport.collide(diff_l2, diff);
      double l2_error = std::sqrt(diff * diff_l2);
      transport.collide(diff_l2, flux_coarsened_g);
      double l2_norm = std::sqrt(flux_coarsened_g * diff_l2);
      if (is_relative)
        l2_error /= l2_norm;
      l2_errors[g] = l2_error;
      table.add_value(key, l2_errors[g]);
    }
    table.set_scientific(key, true);
    table.set_precision(key, 16);
  }
};

#endif  // AETHER_EXAMPLES_COARSE_TEST_H_