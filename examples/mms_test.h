#ifndef AETHER_EXAMPLES_MMS_TEST_H_
#define AETHER_EXAMPLES_MMS_TEST_H_

#include <deal.II/grid/grid_out.h>

#include "example_test.h"

template <int dim, int qdim>
class MmsTest : virtual public ExampleTest<dim, qdim> {
 protected:
  using ExampleTest<dim, qdim>::mesh;
  using ExampleTest<dim, qdim>::quadrature;
  using ExampleTest<dim, qdim>::dof_handler;
  using ExampleTest<dim, qdim>::mgxs;

  template <class Solution>
  void CreateSource(
      dealii::BlockVector<double> &source,
      const std::vector<Solution> &solutions_spaceangle,
      const std::vector<dealii::Vector<double>> &solutions_energy,
      const dealii::DoFHandler<dim> &dof_handler,
      const QAngle<qdim> &quadrature,
      const Mgxs &mgxs) {
    std::vector<dealii::BlockVector<double>> sources_spaceangle;
    std::vector<dealii::Vector<double>> sources_energy;
    CreateSources(sources_spaceangle, sources_energy,
                  solutions_spaceangle, solutions_energy,
                  dof_handler, quadrature, mgxs);
    const int num_sources = sources_spaceangle.size();
    AssertDimension(num_sources, sources_energy.size());
    const int num_groups = mgxs.total.size();
    for (int s = 0; s < num_sources; ++s)
      for (int g = 0; g < num_groups; ++g)
        source.block(g).add(sources_energy[s][g], 
                            sources_spaceangle[s].block(0));
  }

  template <class Solution>
  void CreateSources(
        std::vector<dealii::BlockVector<double>> &sources_spaceangle,
        std::vector<dealii::Vector<double>> &sources_energy,
        const std::vector<Solution> &solutions_spaceangle,
        const std::vector<dealii::Vector<double>> &solutions_energy,
        const dealii::DoFHandler<dim> &dof_handler,
        const QAngle<qdim> &quadrature,
        const Mgxs &mgxs) {
    const int num_materials = mgxs.total[0].size();
    AssertDimension(solutions_spaceangle.size(), solutions_energy.size());
    AssertDimension(sources_spaceangle.size(), 0);
    AssertDimension(sources_energy.size(), 0);
    const int num_groups = mgxs.total.size();
    const int num_solutions = solutions_spaceangle.size();
    const int num_sources_per_solution = 1 + 2 * num_materials;
    const int num_sources = num_solutions * num_sources_per_solution;
    sources_spaceangle.resize(num_sources, 
        dealii::BlockVector<double>(1, quadrature.size()*dof_handler.n_dofs()));
    sources_energy.resize(num_sources, dealii::Vector<double>(num_groups));
    for (int i = 0; i < num_solutions; ++i) {
      const int s = i * num_sources_per_solution;
      dealii::BlockVector<double> source_spaceangle(
          quadrature.size(), dof_handler.n_dofs());
      auto grad = std::bind(&Solution::gradient, solutions_spaceangle[i], 
                            std::placeholders::_1, 0);
      for (int n = 0; n < quadrature.size(); ++n) {
        Streamed<dim> streamed(quadrature.ordinate(n), grad);
        dealii::VectorTools::interpolate(
            dof_handler, streamed, source_spaceangle.block(n));
      }
      sources_spaceangle[s].block(0) = source_spaceangle;
      sources_energy[s] = solutions_energy[i];
      // collision and scattering
      dealii::Vector<double> source_spaceangle_iso(dof_handler.n_dofs());
      dealii::VectorTools::interpolate(
          dof_handler, solutions_spaceangle[i], source_spaceangle_iso);
      for (int j = 0; j < num_materials; ++j) {
        for (int g = 0; g < num_groups; ++g) {
          sources_energy[s+1+j][g] = mgxs.total[g][j] 
                                      * solutions_energy[i][g];
          for (int gp = 0; gp < num_groups; ++gp)
            sources_energy[s+1+num_materials+j][g] += mgxs.scatter[g][gp][j]
                                                      * solutions_energy[i][gp];
        }
        dealii::Vector<double> source_spaceangle_iso_j(source_spaceangle_iso);
        std::vector<dealii::types::global_dof_index> dof_indices(
            dof_handler.get_fe().dofs_per_cell);
        for (auto cell = dof_handler.begin_active();
              cell != dof_handler.end(); ++cell) {
          if (cell->material_id() != j) {
            cell->get_dof_indices(dof_indices);
            for (auto index : dof_indices) {
              source_spaceangle_iso_j[index] = 0;
            }
          }
        }
        source_spaceangle = 0;
        for (int n = 0; n < quadrature.size(); ++n)
          source_spaceangle.block(n) = source_spaceangle_iso_j;
        sources_spaceangle[s+1+j].block(0) = source_spaceangle;
        sources_spaceangle[s+1+num_materials+j].block(0) = source_spaceangle;
        sources_spaceangle[s+1+num_materials+j].block(0) *= -1;
      }
    }
  }

  void TestFullOrder(const int num_cycles,
                     const int max_iters, 
                     const double tol, 
                     const double period) {
    const int num_groups = this->mgxs->total.size();
    const int num_materials = this->mgxs->total[0].size();
    std::vector<std::vector<dealii::BlockVector<double>>> boundary_conditions(
        num_groups, std::vector<dealii::BlockVector<double>>());
    // Create solution
    using Solution = CosineFunction<dim>;
    std::vector<Solution> solutions_spaceangle;
    solutions_spaceangle.emplace_back(period);
    std::vector<dealii::Vector<double>> solutions_energy(1,
        dealii::Vector<double>(num_groups));
    for (int g = 0; g < num_groups; ++g)
      solutions_energy[0][g] = g + 1;
    // Run MMS refinement cycles
    dealii::ConvergenceTable convergence_table;
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
      // Print mesh
      dealii::GridOutFlags::Svg svg;
      svg.coloring = dealii::GridOutFlags::Svg::Coloring::material_id;
      svg.margin = false;
      svg.label_cell_index = false;
      svg.label_level_number = false;
      svg.label_level_subdomain_id = false;
      svg.label_material_id = false;
      svg.label_subdomain_id = false;
      svg.draw_colorbar = false;
      svg.draw_legend = false;
      dealii::GridOut grid_out;
      grid_out.set_flags(svg);
      std::string filename = "mesh_mms"+std::to_string(cycle)+".svg";
      std::ofstream file(filename);
      grid_out.write_svg(mesh, file);
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
      dealii::ReductionControl control(max_iters, 1e-10, tol);
      dealii::SolverGMRES<dealii::BlockVector<double>> solver(control,
          dealii::SolverGMRES<dealii::BlockVector<double>>::AdditionalData(32));
      solver.connect([](const unsigned int iteration,
                        const double check_value,
                        const dealii::BlockVector<double>&) {
        std::cout << iteration << ": " << check_value << std::endl;
        return dealii::SolverControl::success;
      });
      dealii::PreconditionIdentity preconditioner;
      std::cout << "running full-order: cycle " << cycle << std::endl;
      solver.solve(problem.fixed_source, flux, uncollided, preconditioner);
      // Plot solution
      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(this->dof_handler);
      data_out.add_data_vector(flux.block(0), "g1 scalar");
      data_out.build_patches();
      std::string name = this->GetTestName() + "_cycle" + std::to_string(cycle);
      std::ofstream output_vtu(name+".vtu");
      data_out.write_vtu(output_vtu);
      // Post-process
      dealii::BlockVector<double> flux_g(
          quadrature.size(), dof_handler.n_dofs());
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
          double summand = l2_error_sq * quadrature.weight(n);
          l2_errors_g[cycle][g] += summand;
          l2_errors[cycle] += summand;
        }
        l2_errors_g[cycle][g] = std::sqrt(l2_errors_g[cycle][g]);
      }
      l2_errors[cycle] = std::sqrt(l2_errors[cycle]);
      convergence_table.add_value("total", l2_errors[cycle]);
      if (cycle > 0) {
        double ratio = l2_errors[cycle-1]/l2_errors[cycle];
        double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
        EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 1e-1);
        std::cout << l2_conv << std::endl;
      }
    }
    convergence_table.set_scientific("total", true);
    convergence_table.set_precision("total", 16);
    this->WriteConvergenceTable(convergence_table);
  }

  void TestPgd(const int num_cycles, const int max_iters, const double period) {
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
    using Solution = CosineFunction<dim>;
    std::vector<Solution> solutions_spaceangle;
    solutions_spaceangle.emplace_back(period);
    std::vector<dealii::Vector<double>> solutions_energy(1,
        dealii::Vector<double>(num_groups));
    for (int g = 0; g < num_groups; ++g)
      solutions_energy[0][g] = g + 1;
    // Run MMS refinement cycles
    dealii::ConvergenceTable convergence_table;
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
          {&energy_mg, &fixed_source_p};
      pgd::sn::NonlinearGS nonlinear_gs(
          linear_ops, num_materials, 1, num_sources);
      const int num_modes = 1;
      for (int m = 0; m < num_modes; ++m) {
        nonlinear_gs.enrich();
        for (int k = 0; k < max_iters; ++k) {
          double residual = nonlinear_gs.step(dealii::BlockVector<double>(),
                                              dealii::BlockVector<double>());
          // if (residual < tol)
          //   break;
        }
        // nonlinear_gs.finalize();
        // if (m > 0)
        //   nonlinear_gs.update();
      }
      // Post-process
      dealii::BlockVector<double> flux_g(
          quadrature.size(), dof_handler.n_dofs());
      for (int g = 0; g < num_groups; ++g) {
        flux_g = fixed_source_p.caches[0].mode.block(0);
        flux_g *= energy_mg.modes[0][g];
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
          double summand = l2_error_sq * quadrature.weight(n);
          l2_errors_g[cycle][g] += summand;
          l2_errors[cycle] += summand;
        }
        l2_errors_g[cycle][g] = std::sqrt(l2_errors_g[cycle][g]);
      }
      l2_errors[cycle] = std::sqrt(l2_errors[cycle]);
      convergence_table.add_value("total", l2_errors[cycle]);
      if (cycle > 0) {
        double ratio = l2_errors[cycle-1]/l2_errors[cycle];
        double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
        EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 1e-1);
        std::cout << l2_conv << std::endl;
      }
    }
    convergence_table.set_scientific("total", true);
    convergence_table.set_precision("total", 16);
    this->WriteConvergenceTable(convergence_table);
  }
};

#endif  // AETHER_EXAMPLES_MMS_TEST_H_