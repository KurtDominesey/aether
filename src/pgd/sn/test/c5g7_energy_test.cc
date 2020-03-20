#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include "mesh/mesh.h"
#include "sn/quadrature.h"
#include "sn/transport.h"
#include "sn/transport_block.h"
#include "sn/quadrature.h"
#include "sn/quadrature_lib.h"
#include "base/mgxs.cc"
#include "base/lapack_full_matrix.h"
#include "functions/function_lib.h"
#include "sn/fixed_source_problem.cc"
#include "sn/fixed_source_gs.h"
#include "pgd/sn/energy_mg_full.h"
#include "pgd/sn/fixed_source_p.h"
#include "pgd/sn/nonlinear_gs.h"

#include "gtest/gtest.h"

namespace aether {

namespace {

using namespace aether::sn;

class C5G7EnergyTest : public ::testing::Test {
 protected:
  static const int dim = 2;
  static const int qdim = 2;
  // static const int num_mats = 2;
  // const int num_groups = 7;
  const int num_ell = 1;
  const double pitch = 0.63;
  const double radius = 0.54;
  dealii::Triangulation<dim> mesh;

  Mgxs ReadMgxs() {
    const std::string filename = "/mnt/c/Users/kurt/Documents/projects/aether/examples/c5g7/c5g7.h5";
    const std::string temperature = "294K";
    const std::vector<std::string> materials =  {"water", "uo2", "void"};
    Mgxs mgxs = read_mgxs(filename, temperature, materials);
    const int num_groups = mgxs.total.size();
    for (int g = 0; g < num_groups; ++g) {
      for (int gp = 0; gp < num_groups; ++gp) {
        for (int j = 0; j < materials.size(); ++j) {
          std::cout << mgxs.scatter[g][gp][j] << " ";
        }
      }
      std::cout << std::endl;
    }
    return mgxs;
  }

  Mgxs ReadMgxsCladded(const std::string &group_structure) {
    const std::string filename = 
        "/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/mgxs-"
        + group_structure + ".h5";
    const std::string temperature = "294K";
    const std::vector<std::string> materials = 
        {"void", "water", "uo2", "zr", "al"};
    Mgxs mgxs = read_mgxs(filename, temperature, materials);
    return mgxs;
  }

  void MeshQuarterPincell() {
    mesh_quarter_pincell(mesh, {radius}, pitch, {1, 0});
    TransformMesh();
  }

  void MeshQuarterPincellCladded() {
    std::vector<double> radii{0.4095, 0.4180, 0.4750, 0.4850, 0.5400};
    mesh_quarter_pincell(mesh, radii, pitch, {2, 0, 3, 0, 4, 1});
    TransformMesh();
  }

  void MeshEighthPincell() {
    mesh_eighth_pincell(mesh, {radius}, pitch, {1, 0});
    TransformMesh();
    // (Set reflecting boundary)
    using Cell = typename dealii::Triangulation<dim>::active_cell_iterator;
    using Face = typename dealii::Triangulation<dim>::active_face_iterator;
    for (Cell cell = mesh.begin_active(); cell != mesh.end(); ++cell) {
      for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
        Face face = cell->face(f);
        if (face->at_boundary()) {
          bool at_diag = true;
          for (int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_face; ++v)
            at_diag = at_diag && (face->vertex(v)[0] == face->vertex(v)[1]);
          if (at_diag)
            face->set_boundary_id(types::reflecting_boundary_id);
        }
      }
    }
  }

  void TransformMesh() {
    // (Set reflecting boundary)
    using Cell = typename dealii::Triangulation<dim>::active_cell_iterator;
    using Face = typename dealii::Triangulation<dim>::active_face_iterator;
    for (Cell cell = mesh.begin_active(); cell != mesh.end(); ++cell) {
      for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
        Face face = cell->face(f);
        if (face->at_boundary()) {
          face->set_boundary_id(types::reflecting_boundary_id);
        }
      }
    }
    // Refine mesh
    mesh.refine_global(2);
  }

  void TestFullOrder(const Mgxs &mgxs) {
    const int num_groups = mgxs.total.size();
    const int num_materials = mgxs.total[0].size();
    // Create quadrature
    const QPglc<dim, qdim> quadrature(1, 2);
    // Create finite elements
    dealii::FE_DGQ<dim> fe(1);
    dealii::DoFHandler dof_handler(mesh);
    dof_handler.distribute_dofs(fe);
    // Create boundary conditions
    std::vector<std::vector<dealii::BlockVector<double>>> boundary_conditions(
        num_groups, std::vector<dealii::BlockVector<double>>(
            0, dealii::BlockVector<double>(quadrature.size(),
                                           fe.dofs_per_cell)));
    // Create solution
    double factor = dealii::numbers::PI_2 / pitch;
    using Solution = CosineFunction<dim>;
    Solution solution(factor * 1);
    dealii::Vector<double> solution_energy(num_groups);
    for (int g = 0; g < num_groups; ++g)
      solution_energy[g] = g + 1;
    // Run MMS refinement cycles
    dealii::ConvergenceTable convergence_table;
    int num_cycles = 5;
    double l2_errors[num_cycles][num_groups][quadrature.size()];
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
      if (cycle > 0) {
        mesh.refine_global();
        dof_handler.initialize(mesh, dof_handler.get_fe());
      }
      // Create source
      dealii::BlockVector<double> source(
          num_groups, quadrature.size() * dof_handler.n_dofs());
      for (int g = 0; g < num_groups; ++g) {
        dealii::BlockVector<double> source_g(quadrature.size(), 
                                             dof_handler.n_dofs());
        for (int n = 0; n < quadrature.size(); ++n) {
          auto grad = std::bind(
              &Solution::gradient, solution, std::placeholders::_1, 0);
          Streamed<dim> streamed(quadrature.ordinate(n), grad);
          dealii::VectorTools::interpolate(
              dof_handler, streamed, source_g.block(n));
          source_g.block(n) *= solution_energy[g];
        }
        dealii::Vector<double> source_iso(dof_handler.n_dofs());
        dealii::VectorTools::interpolate(dof_handler, solution, source_iso);
        dealii::Vector<double> source_collision(source_iso);
        dealii::Vector<double> source_scattering(source_iso);
        std::vector<double> collision(num_materials);
        std::vector<double> scattering(num_materials);
        for (int j = 0; j < num_materials; ++j) {
          collision[j] = solution_energy[g] * mgxs.total[g][j];
          for (int gp = 0; gp < num_groups; ++gp)
            scattering[j] += solution_energy[gp] * mgxs.scatter[g][gp][j];
        }
        std::vector<dealii::types::global_dof_index> dof_indices(
            fe.dofs_per_cell);
        for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
             ++cell) {
          const int j = cell->material_id();
          cell->get_dof_indices(dof_indices);
          for (auto index : dof_indices) {
            source_collision[index] *= collision[j];
            source_scattering[index] *= -scattering[j];
          }
        }
        for (int n = 0; n < quadrature.size(); ++n) {
          source_g.block(n) += source_collision;
          source_g.block(n) += source_scattering;
        }
        source.block(g) = source_g;
      }
      // Run problem
      FixedSourceProblem<dim, qdim> problem(
          dof_handler, quadrature, mgxs, boundary_conditions);
      dealii::BlockVector<double> uncollided(source.get_block_indices());
      dealii::BlockVector<double> flux(source.get_block_indices());
      problem.sweep_source(uncollided, source);
      dealii::SolverControl control(3000, 1e-8);
      dealii::SolverRichardson<dealii::BlockVector<double>> solver(control);
      solver.solve(problem.fixed_source, flux, uncollided, 
                   dealii::PreconditionIdentity());
      // Verify solution
      dealii::BlockVector<double> flux_g(quadrature.size(), 
                                         dof_handler.n_dofs());
      for (int g = 0; g < num_groups; ++g) {
        flux_g = flux.block(g);
        flux_g /= solution_energy[g];
        for (int n = 0; n < quadrature.size(); ++n) {
          dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
          dealii::VectorTools::integrate_difference(
              dof_handler, flux_g.block(n), solution, 
              difference_per_cell,
              dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
              dealii::VectorTools::L2_norm);
          double l2_error = difference_per_cell.l2_norm() * solution_energy[g];
          l2_errors[cycle][g][n] = l2_error;
          if (cycle > 0) {
            double ratio = l2_errors[cycle-1][g][n] / l2_errors[cycle][g][n];
            double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
            EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 5e-2);
          }
          // std::string key = "g" + std::to_string(g) + "n" + std::to_string(n);
          // convergence_table.add_value(key, l2_error);
          // convergence_table.set_scientific(key, true);
        }
        double l2_error_g = 0;
        for (int n = 0; n < quadrature.size(); ++n)
          l2_error_g += quadrature.weight(n) *
                        std::pow(l2_errors[cycle][g][n], 2);
        l2_error_g = std::sqrt(l2_error_g);
        std::string key = "g" + std::to_string(g);
        convergence_table.add_value(key, l2_error_g);
        convergence_table.set_scientific(key, true);
      }
      if (cycle == num_cycles - 1) {
        // Output results
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        std::vector<dealii::BlockVector<double>*> flux_groups;
        for (int g = 0; g < num_groups; ++g) {
          auto flux_g = new dealii::BlockVector<double>(
              quadrature.size(), dof_handler.n_dofs());
          flux_groups.push_back(flux_g);
          (*flux_g) = flux.block(g);
          for (int n = 0; n < quadrature.size(); ++n) {
            data_out.add_data_vector(
                flux_g->block(n), 
                "g" + std::to_string(g) + "n" + std::to_string(n),
                dealii::DataOut_DoFData<dealii::DoFHandler<dim>,
                                        dim>::DataVectorType::type_dof_data);
          }
        }
        data_out.build_patches();
        std::ofstream output("flux_full_order.vtu");
        data_out.write_vtu(output);
        for (dealii::BlockVector<double>* flux_g : flux_groups)
          delete flux_g;
      }
    }
    convergence_table.evaluate_all_convergence_rates(
        dealii::ConvergenceTable::RateMode::reduction_rate_log2);
    std::ofstream table_out("convergence_full.txt");
    convergence_table.write_text(table_out);
    convergence_table.write_tex(std::cout);
  }

  void Test(const Mgxs &mgxs) {
    const int num_groups = mgxs.total.size();
    const int num_materials = mgxs.total[0].size();
    Mgxs mgxs_one(1, num_materials, num_ell);
    Mgxs mgxs_pseudo(1, num_materials, num_ell);
    for (int j = 0; j < num_materials; ++j) {
      mgxs_one.total[0][j] = 1;
      mgxs_one.scatter[0][0][j] = 1;
    }
    // Create quadrature
    const QPglc<dim, qdim> quadrature(1, 2);
    // Create finite elements
    dealii::FE_DGQ<dim> fe(1);
    dealii::DoFHandler dof_handler(mesh);
    dof_handler.distribute_dofs(fe);
    // Create boundary conditions
    std::vector<std::vector<dealii::BlockVector<double>>> boundary_conditions(
        1, std::vector<dealii::BlockVector<double>>(
            0, dealii::BlockVector<double>(quadrature.size(),
                                           fe.dofs_per_cell)));
    // Create solutions
    using Solution = CosineFunction<dim>;
    const int num_solutions = 1;
    std::vector<Solution> solutions_spaceangle;
    std::vector<dealii::Vector<double>> solutions_energy(
        num_solutions, dealii::Vector<double>(num_groups));
    double factor = dealii::numbers::PI_2 / pitch;
    for (int i = 0; i < num_solutions; ++i) {
      for (int g = 0; g < num_groups; ++g)
        solutions_energy[i][g] = g + 1;
      solutions_spaceangle.emplace_back(factor * 4);
    }
    // Run MMS refinement cycles
    dealii::ConvergenceTable convergence_table;
    int num_cycles = 2;
    double l2_errors[num_cycles][quadrature.size()];
    double l2_errors_energy[num_cycles];
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
      if (cycle > 0) {
        mesh.refine_global();
        dof_handler.initialize(mesh, dof_handler.get_fe());
      }
      std::vector<dealii::BlockVector<double>> sources_spaceangle;
      std::vector<dealii::Vector<double>> sources_energy;
      CreateSources(sources_spaceangle, sources_energy,
                    solutions_spaceangle, solutions_energy,
                    dof_handler, quadrature, mgxs);
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
      pgd::sn::EnergyMgFull energy_mg(mgxs, sources_energy);
      std::vector<pgd::sn::LinearInterface*> linear_ops = 
          {&fixed_source_p, &energy_mg};
      pgd::sn::NonlinearGS nonlinear_gs(
          linear_ops, num_materials, num_ell, num_sources);
      nonlinear_gs.enrich();
      for (int k = 0; k < 40; ++k) {
        nonlinear_gs.step(dealii::BlockVector<double>(),
                          dealii::BlockVector<double>());
      }
      // Verify energetic mode
      double scaling = solutions_energy[0].l2_norm();
      energy_mg.modes.back() *= scaling;
      std::cout << "FINAL ENERGY MODE\n";
      energy_mg.modes.back().print(std::cout);
      for (int g = 0; g < num_groups; ++g)
        EXPECT_NEAR(energy_mg.modes.back()[g], solutions_energy[0][g], 
                    solutions_energy[0][g] * 5e-2);
      dealii::Vector<double> error_energy(energy_mg.modes.back());
      error_energy -= solutions_energy[0];
      l2_errors_energy[cycle] = error_energy.l2_norm();
      if (cycle > 0) {
        double ratio = l2_errors_energy[cycle-1] / l2_errors_energy[cycle];
        double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
        std::cout << "L2 ENERGY " << l2_conv << std::endl;
      }
      // Verify spatio-angular mode
      dealii::Vector<double> mode_unrolled(
          fixed_source_p.caches.back().mode.block(0));
      dealii::BlockVector<double> mode_spaceangle(quadrature.size(), 
                                                  dof_handler.n_dofs());
      mode_spaceangle = mode_unrolled;
      mode_spaceangle /= scaling;
      for (int n = 0; n < quadrature.size(); ++n) {
        dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
        dealii::VectorTools::integrate_difference(
            dof_handler, mode_spaceangle.block(n), solutions_spaceangle[0], 
            difference_per_cell,
            dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
            dealii::VectorTools::L2_norm);
        double l2_error = difference_per_cell.l2_norm();
        l2_errors[cycle][n] = l2_error;
        if (cycle > 0) {
          double ratio = l2_errors[cycle-1][n] / l2_errors[cycle][n];
          double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
          EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 5e-2);
        }
        // std::string key = "L2 " + std::to_string(n);
        // convergence_table.add_value(key, l2_error);
        // convergence_table.set_scientific(key, true);
      }
      double l2_error_m = 0;
      for (int n = 0; n < quadrature.size(); ++n)
        l2_error_m += quadrature.weight(n) * std::pow(l2_errors[cycle][n], 2);
      l2_error_m = std::sqrt(l2_error_m);
      std::string key_psi = "L2 psi";
      convergence_table.add_value(key_psi, l2_error_m);
      convergence_table.set_scientific(key_psi, true);
      std::string key_energy = "L2 energy";
      convergence_table.add_value(key_energy, l2_errors_energy[cycle]);
      convergence_table.set_scientific(key_energy, true);
      if (cycle == num_cycles - 1) {
        // Output results
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        for (int n = 0; n < quadrature.size(); ++n) {
          data_out.add_data_vector(
              mode_spaceangle.block(n), "n" + std::to_string(n),
              dealii::DataOut_DoFData<dealii::DoFHandler<dim>,
                                      dim>::DataVectorType::type_dof_data);
        }
        data_out.build_patches();
        std::ofstream output("flux.vtu");
        data_out.write_vtu(output);
      }
    }
    convergence_table.evaluate_all_convergence_rates(
      dealii::ConvergenceTable::RateMode::reduction_rate_log2);
    std::ofstream table_out("convergence_pgd.txt");
    convergence_table.write_text(table_out);
    convergence_table.write_tex(std::cout);
  }

void TestMultiple(const Mgxs &mgxs) {
    const int num_groups = mgxs.total.size();
    const int num_materials = mgxs.total[0].size();
    Mgxs mgxs_one(1, num_materials, num_ell);
    Mgxs mgxs_pseudo(1, num_materials, num_ell);
    for (int j = 0; j < num_materials; ++j) {
      mgxs_one.total[0][j] = 1;
      mgxs_one.scatter[0][0][j] = 1;
    }
    // Create quadrature
    const QPglc<dim, qdim> quadrature(1, 2);
    // Create finite elements
    dealii::FE_DGQ<dim> fe(1);
    dealii::DoFHandler dof_handler(mesh);
    dof_handler.distribute_dofs(fe);
    // Create boundary conditions
    std::vector<std::vector<dealii::BlockVector<double>>> boundary_conditions(
        1, std::vector<dealii::BlockVector<double>>());
    // Create solutions
    using Solution = CosineFunction<dim>;
    const int num_solutions = 2;
    std::vector<Solution> solutions_spaceangle;
    std::vector<dealii::Vector<double>> solutions_energy(
        num_solutions, dealii::Vector<double>(num_groups));
    double factor = dealii::numbers::PI_2 / pitch;
    for (int i = 0; i < num_solutions; ++i) {
      solutions_spaceangle.emplace_back(factor * (i + 1));
    }
    for (int g = 0; g < num_groups; ++g) {
      solutions_energy[0][g] = g * 1e-1;
      solutions_energy[1][g] = (num_groups - 1 - g) * 1;
    }
    // Run MMS refinement cycles
    dealii::ConvergenceTable convergence_table;
    int num_cycles = 2;
    double l2_errors[num_cycles][num_groups][quadrature.size()];
    dealii::DataOut<dim> data_out;
    std::vector<dealii::BlockVector<double>*> flux_groups;
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
      if (cycle > 0) {
        mesh.refine_global();
        dof_handler.initialize(mesh, dof_handler.get_fe());
      }
      std::vector<dealii::BlockVector<double>> sources_spaceangle;
      std::vector<dealii::Vector<double>> sources_energy;
      std::cout << "making sources\n";
      CreateSources(sources_spaceangle, sources_energy,
                    solutions_spaceangle, solutions_energy,
                    dof_handler, quadrature, mgxs);
      std::cout << "made sources\n";
      const int num_sources = sources_spaceangle.size();
      using TransportType = pgd::sn::Transport<dim, qdim>;
      using TransportBlockType = pgd::sn::TransportBlock<dim, qdim>;
      FixedSourceProblem<dim, qdim, TransportType, TransportBlockType> problem(
          dof_handler, quadrature, mgxs_pseudo, boundary_conditions);
      pgd::sn::FixedSourceP fixed_source_p(
          problem.fixed_source, mgxs_pseudo, mgxs_one, sources_spaceangle);
      pgd::sn::EnergyMgFull energy_mg(mgxs, sources_energy);
      std::vector<pgd::sn::LinearInterface*> linear_ops = 
          {&fixed_source_p, &energy_mg};
      pgd::sn::NonlinearGS nonlinear_gs(
          linear_ops, num_materials, num_ell, num_sources);
      const int num_modes = 5;  //num_solutions;
      for (int m = 0; m < num_modes; ++m) {
        nonlinear_gs.enrich();
        if (m < num_solutions) {
          energy_mg.modes[m] = solutions_energy[num_solutions-1-m];
          nonlinear_gs.set_inner_products();
        }
        const int num_iters = 150;
        for (int k = 0; k < num_iters; ++k) {
          nonlinear_gs.step(dealii::BlockVector<double>(),
                            dealii::BlockVector<double>(),
                            k < num_iters - 1);
        }
      }
      // Verify energetic mode
      // double scaling = solutions_energy[0].l2_norm();
      // energy_mg.modes.back() *= scaling;
      for (int m = 0; m < num_modes; ++m) {
        dealii::Vector<double> mode(energy_mg.modes[m]);
        std::cout << "MAGNITUDE " 
                  << fixed_source_p.caches[m].mode.l2_norm() * mode.l2_norm()
                  << std::endl;
        // mode /= mode.l2_norm();
        // mode *= solutions_energy[m].l2_norm();
        std::cout << "ENERGY MODE\n";
        mode.print(std::cout);
      }
      // energy_mg.modes.back().print(std::cout);
      // Evaluate tensor product
      dealii::BlockVector<double> flux(
          num_groups, quadrature.size() * dof_handler.n_dofs());
      dealii::BlockVector<double> flux_g(
          quadrature.size(), dof_handler.n_dofs());
      for (int g = 0; g < num_groups; ++g) {
        for (int m = 0; m < num_modes; ++m) {
          flux.block(g).add(energy_mg.modes[m][g], 
                            fixed_source_p.caches[m].mode.block(0));
        }
        flux_g = flux.block(g);
        for (int n = 0; n < quadrature.size(); ++n) {
          dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
          dealii::VectorTools::integrate_difference(
              dof_handler, flux_g.block(n), solutions_spaceangle[0], 
              difference_per_cell,
              dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
              dealii::VectorTools::L2_norm);
          double l2_error = difference_per_cell.l2_norm();
          l2_errors[cycle][g][n] = l2_error;
          if (cycle > 0) {
            double ratio = l2_errors[cycle-1][g][n] / l2_errors[cycle][g][n];
            double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
            EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 5e-2);
          }
        }
        // Output results
        if (cycle == num_cycles - 1) {
          data_out.attach_dof_handler(dof_handler);
          auto flux_group = new dealii::BlockVector<double>(flux_g);
          flux_groups.push_back(flux_group);
          for (int n = 0; n < quadrature.size(); ++n) {
            std::string key = "g" + std::to_string(g) + "n" + std::to_string(n);
            data_out.add_data_vector(flux_group->block(n), key);
          }
        }
      }
    }
    data_out.build_patches();
    std::ofstream output("flux.vtu");
    data_out.write_vtu(output);
    for (auto flux_group : flux_groups)
      delete flux_group;
    convergence_table.evaluate_all_convergence_rates(
      dealii::ConvergenceTable::RateMode::reduction_rate_log2);
    std::ofstream table_out("convergence_pgd.txt");
    convergence_table.write_text(table_out);
    convergence_table.write_tex(std::cout);
  }

  template <int dim, int qdim, class Solution>
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
        // double collision = 0;
        // double scattering = 0;
        // for (int g = 0; g < num_groups; ++g) {
        //   collision += solutions_energy[i][g] * sources_energy[i+1+j][g];
        //   scattering += solutions_energy[i][g] 
        //                 * sources_energy[i+1+num_mats+j][g];
        // }
        // collision /= std::pow(solutions_energy[i].l2_norm(), 2);
        // scattering /= std::pow(solutions_energy[i].l2_norm(), 2);
        // std::cout << "COLLISION TRUE " << collision << std::endl;
        // std::cout << "SCATTERING TRUE " << scattering << std::endl;
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

  void Compare(const Mgxs &mgxs) {
    // Create quadrature
    const QPglc<dim, qdim> quadrature(1, 2);
    // Create finite elements
    dealii::FE_DGQ<dim> fe(1);
    dealii::DoFHandler dof_handler(mesh);
    dof_handler.distribute_dofs(fe);
    // Create sources
    const int num_groups = mgxs.total.size();
    const int num_materials = mgxs.total[0].size();
    const int num_sources = num_materials;
    // AssertDimension(num_materials, 2);
    std::vector<dealii::Vector<double>> sources_energy(
        num_sources, dealii::Vector<double>(num_groups));
    std::vector<dealii::BlockVector<double>> sources_spaceangle(num_sources, 
        dealii::BlockVector<double>(1, quadrature.size()*dof_handler.n_dofs()));
    // (Energy dependence)
    for (int g = 0; g < num_groups; ++g) {
      for (int j = 0; j < mgxs.chi[g].size(); ++j) {
        sources_energy[j][g] = mgxs.chi[g][j];
      }
    }
    // (Spatio-angular dependence)
    std::cout << "(Spatio-angular dependence)\n";
    std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); 
         ++cell) {
      cell->get_dof_indices(dof_indices);
      const int j = cell->material_id();
      for (int n = 0; n < quadrature.size(); ++n) {
        for (dealii::types::global_dof_index i : dof_indices) {
          sources_spaceangle[j][n*dof_handler.n_dofs() + i] = 1.0;
        }
      }
    }
    // Create boundary conditions
    std::cout << "Create boundary conditions\n";
    std::vector<dealii::BlockVector<double>> boundary_conditions_one;
    std::vector<std::vector<dealii::BlockVector<double>>> 
        boundary_conditions(num_groups);
    // Set group structure
    const std::vector<int> g_maxes = {10, 14, 18, 24, 43, 55, 65, 70};
    // Run full order
    dealii::BlockVector<double> source_full(
        num_groups, quadrature.size() * dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g)
      for (int j = 0; j < num_sources; ++j)
        source_full.block(g).add(
            sources_energy[j][g], sources_spaceangle[j].block(0));
    dealii::BlockVector<double> flux_full(source_full.get_block_indices());
    FixedSourceProblem<dim, qdim> problem_full(
        dof_handler, quadrature, mgxs, boundary_conditions);
    RunFullOrder(flux_full, source_full, problem_full);
    dealii::BlockVector<double> flux_full_iso(num_groups, dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g)
      problem_full.d2m.vmult(flux_full_iso.block(g), flux_full.block(g));
    Mgxs mgxs_coarse = collapse_mgxs(
        flux_full_iso, dof_handler, problem_full.transport, mgxs, g_maxes);
    const std::string temperature = "294K";
    const std::vector<std::string> materials = 
        {"void", "water", "uo2", "zr", "al"};
    write_mgxs(mgxs_coarse, "mgxs_coarse.h5", temperature, materials);
    return;
    // RunFullOrder(flux_full, source_full, boundary_conditions, 
    //              dof_handler, quadrature, mgxs);
    // Run pgd
    Mgxs mgxs_one(1, num_materials, num_ell);
    Mgxs mgxs_pseudo(1, num_materials, num_ell);
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
    pgd::sn::EnergyMgFull energy_mg(mgxs, sources_energy);
    std::vector<pgd::sn::LinearInterface*> linear_ops = 
        {&fixed_source_p, &energy_mg};
    pgd::sn::NonlinearGS nonlinear_gs(
        linear_ops, num_materials, num_ell, num_sources);
    const double nonlinear_tol = 1e-6;
    const int num_modes = 20;  //num_solutions;
    for (int m = 0; m < num_modes; ++m) {
      nonlinear_gs.enrich();
      const int num_iters = 50;
      double res = std::nan("a");
      double res_prev = std::nan("b");
      for (int k = 0; k < num_iters; ++k) {
        res_prev = res;
        res = nonlinear_gs.step(dealii::BlockVector<double>(),
                                dealii::BlockVector<double>(),
                                k < num_iters - 1);
        std::cout << "nonlinear residual " << res << std::endl;
        bool diverging = k >= 1 && (res - res_prev) > (res_prev / 10.);
        if (res < nonlinear_tol || diverging)
          break;
      }
      std::cout << "finalize\n";
      nonlinear_gs.finalize();
      std::cout << "update\n";
      nonlinear_gs.update();
    }
    dealii::BlockVector<double> source(flux_full.get_block_indices());
    dealii::BlockVector<double> uncollided(flux_full.get_block_indices());
    dealii::BlockVector<double> operated(flux_full.get_block_indices());
    for (int i = 0; i < num_sources; ++i)
      for (int g = 0; g < num_groups; ++g)
        source.block(g).add(sources_energy[i][g], 
                            sources_spaceangle[i].block(0));
    problem_full.sweep_source(uncollided, source);
    // Get SVD of full-order solution
    // LAPACKFullMatrix<double> flux_full_matrix(
    //     num_groups, quadrature.size() * dof_handler.n_dofs());
    // for (int g = 0; g < num_groups; ++g) {
    //   for (int n = 0; n < quadrature.size(); ++n) {
    //     for (int i = 0; i < dof_handler.n_dofs(); ++i) {
    //       flux_full_matrix(g, n*dof_handler.n_dofs() + i) =
    //           flux_full.block(g)[n*dof_handler.n_dofs() + i];
    //     }
    //   }
    // }
    // flux_full_matrix.compute_svd();
    // const dealii::LAPACKFullMatrix<double> svd_spaceangle =
    //     flux_full_matrix.get_svd_u();
    // const dealii::LAPACKFullMatrix<double> svd_energy =
    //     flux_full_matrix.get_svd_vt();
    // std::vector<dealii::BlockVector<double>> svd_vectors_spaceangle(num_groups, 
    //     dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
    // std::vector<dealii::Vector<double>> svd_vectors_energy(num_groups,
    //     dealii::Vector<double>(num_groups));
    // for (int g = 0; g < num_groups; ++g) {
    //   svd_vectors_energy[g] = 
    // }
    dealii::BlockVector<double> diff(num_groups, dof_handler.n_dofs());
    dealii::BlockVector<double> diff_g(
        quadrature.size(), dof_handler.n_dofs());
    dealii::BlockVector<double> flux_g(diff_g.get_block_indices());
    dealii::BlockVector<double> flux_iso_g(1, dof_handler.n_dofs());
    dealii::BlockVector<double> diff_iso_g(1, dof_handler.n_dofs());
    dealii::Vector<double> diff_l2(dof_handler.n_dofs());
    dealii::BlockVector<double> flux_pgd(flux_full.get_block_indices());
    dealii::BlockVector<double> moments_pgd(num_groups, 1*dof_handler.n_dofs());
    const int num_ordinates = quadrature.size();
    std::vector<double> l2_errors(num_modes+1);
    std::vector<std::vector<double>> l2_errors_g(
        num_modes+1, std::vector<double>(num_groups));
    std::vector<std::vector<std::vector<double>>> l2_errors_n(
        num_modes+1, std::vector<std::vector<double>>(
            num_groups, std::vector<double>(num_ordinates)));
    auto l2_errors_iso_g(l2_errors_g);
    auto l2_errors_iso(l2_errors);
    auto l2_residuals_n(l2_errors_n);
    auto l2_residuals_g(l2_errors_g);
    auto l2_residuals(l2_errors);
    std::vector<double> l2_norms(num_modes+1);
    for (int m = 0; m <= num_modes; ++m) {
      for (int g = 0; g < num_groups; ++g) {
        if (m > 0) {
          flux_pgd.block(g).add(energy_mg.modes[m-1][g], 
                                fixed_source_p.caches[m-1].mode.block(0));
          double l2_norm = 0;
          dealii::BlockVector<double> mode(
              quadrature.size(), dof_handler.n_dofs());
          mode = fixed_source_p.caches[m-1].mode.block(0);
          for (int n = 0; n < quadrature.size(); ++n) {
            problem.transport.collide(diff_l2, mode.block(n));
            l2_norm += quadrature.weight(n) * (mode.block(n) * diff_l2);
          }
          double sum_energy = 0;
          for (int g = 0; g < num_groups; ++g)
            sum_energy += std::pow(energy_mg.modes[m-1][g], 2);
          l2_norm = std::sqrt(l2_norm * sum_energy);
          l2_norms[m] = l2_norm;
        }
        problem.d2m.vmult(moments_pgd.block(g), flux_pgd.block(g));
        diff_g = flux_full.block(g);
        flux_g = flux_pgd.block(g);
        problem.d2m.vmult(diff_iso_g, diff_g);
        problem.d2m.vmult(flux_iso_g, flux_g);
        diff_g -= flux_g;
        diff_iso_g -= flux_iso_g;
        for (int n = 0; n < quadrature.size(); ++n) {
          problem.transport.collide(diff_l2, diff_g.block(n));
          l2_errors_n[m][g][n] = std::sqrt(diff_g.block(n) * diff_l2);
          if (m == num_modes)
            diff.block(g) = diff_iso_g.block(0);
            // diff.block(g).add(quadrature.weight(n), diff_g.block(n));
        }
        problem.transport.collide(diff_l2, diff_iso_g.block(0));
        l2_errors_iso_g[m][g] = std::sqrt(diff_iso_g.block(0) * diff_l2);
      }
      Mgxs mgxs_coarse = collapse_mgxs(
          moments_pgd, dof_handler, problem.transport, mgxs, g_maxes);
      problem_full.fixed_source.vmult(operated, flux_pgd);
      operated.sadd(-1, 1, uncollided);  // residual
      for (int g = 0; g < num_groups; ++g) {
        dealii::BlockVector<double> residual_g(
            quadrature.size(), dof_handler.n_dofs());
        residual_g = operated.block(g);
        for (int n = 0; n < quadrature.size(); ++n) {
          dealii::Vector<double> residual_l2(dof_handler.n_dofs());
          problem.transport.collide(residual_l2, residual_g.block(n));
          l2_residuals_n[m][g][n] = std::sqrt(residual_g.block(n)*residual_l2);
        }
      }
    }
    for (int m = 0; m <= num_modes; ++m) {
      for (int g = 0; g < num_groups; ++g) {
        for (int n = 0; n < quadrature.size(); ++n) {
          l2_errors_g[m][g] += quadrature.weight(n) *
                               std::pow(l2_errors_n[m][g][n], 2);
          l2_residuals_g[m][g] += quadrature.weight(n) *
                                  std::pow(l2_residuals_n[m][g][n], 2);
        }
        l2_errors[m] += l2_errors_g[m][g];
        l2_errors_g[m][g] = std::sqrt(l2_errors_g[m][g]);
        l2_residuals[m] += l2_residuals_g[m][g];
        l2_residuals_g[m][g] = std::sqrt(l2_residuals_g[m][g]);
        l2_errors_iso[m] += std::pow(l2_errors_iso_g[m][g], 2);
      }
      l2_errors[m] = std::sqrt(l2_errors[m]);
      l2_residuals[m] = std::sqrt(l2_residuals[m]);
      l2_errors_iso[m] = std::sqrt(l2_errors_iso[m]);
    }
    dealii::ConvergenceTable convergence_table;
    // for (int g = 0; g < num_groups; ++g) {
    //   std::string key = "g" + std::to_string(g);
    //   for (int m = 0; m <= num_modes; ++m)
    //     convergence_table.add_value(key, l2_errors_g[m][g]);
    //   convergence_table.set_scientific(key, true);
    // }
    for (int m = 0; m <= num_modes; ++m) {
      convergence_table.add_value("error", l2_errors[m]);
      convergence_table.add_value("error_iso", l2_errors_iso[m]);
      convergence_table.add_value("norm", l2_norms[m]);
      convergence_table.add_value("residual", l2_residuals[m]);
    }
    convergence_table.set_scientific("error", true);
    convergence_table.set_scientific("error_iso", true);
    convergence_table.set_scientific("norm", true);
    convergence_table.set_scientific("residual", true);
    std::ofstream table_out("convergence_compare.txt");
    convergence_table.write_text(table_out);
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    for (int g = 0; g < num_groups; ++g) {
      diff.block(g).ratio(diff.block(g), flux_full_iso.block(g));
      int pad = std::to_string(num_groups).size() - std::to_string(g).size();
      std::string g_pad = std::string(pad, '0') + std::to_string(g);
      data_out.add_data_vector(diff.block(g), "diff_g"+g_pad);
      data_out.add_data_vector(flux_full_iso.block(g), "flux_g"+g_pad);
    }
    data_out.build_patches();
    std::ofstream output("diff.vtu");
    data_out.write_vtu(output);
  }

  template <int dim, int qdim>
  void RunFullOrder(dealii::BlockVector<double> &flux,
                    const dealii::BlockVector<double> &source,
                    const FixedSourceProblem<dim, qdim> &problem) {
    dealii::BlockVector<double> uncollided(source.get_block_indices());
    problem.sweep_source(uncollided, source);
    dealii::SolverControl control(3000, 1e-8);
    dealii::SolverRichardson<dealii::BlockVector<double>> solver(control);
    solver.solve(problem.fixed_source, flux, uncollided, 
                 dealii::PreconditionIdentity());
  }

  void RunPgd() {}
};

TEST_F(C5G7EnergyTest, MmsPgdQuarter) {
  Mgxs mgxs = ReadMgxs();
  MeshQuarterPincell();
  Test(mgxs);
}

TEST_F(C5G7EnergyTest, MmsPgdEighth) {
  Mgxs mgxs = ReadMgxs();
  MeshEighthPincell();
  Test(mgxs);
}

TEST_F(C5G7EnergyTest, MmsPgdTwoModeQuarter) {
  Mgxs mgxs = ReadMgxs();
  MeshQuarterPincell();
  TestMultiple(mgxs);
}

TEST_F(C5G7EnergyTest, MmsPgdTwoModeEighth) {
  Mgxs mgxs = ReadMgxs();
  MeshEighthPincell();
  TestMultiple(mgxs);
}

TEST_F(C5G7EnergyTest, MmsFullQuarter) {
  Mgxs mgxs = ReadMgxs();
  MeshQuarterPincell();
  TestFullOrder(mgxs);
}

TEST_F(C5G7EnergyTest, MmsFullEighth) {
  Mgxs mgxs = ReadMgxs();
  MeshEighthPincell();
  TestFullOrder(mgxs);
}

TEST_F(C5G7EnergyTest, CompareQuarter) {
  Mgxs mgxs = ReadMgxs();
  MeshQuarterPincell();
  Compare(mgxs);
}

TEST_F(C5G7EnergyTest, CompareQuarterCladded) {
  Mgxs mgxs = ReadMgxsCladded("CASMO-70");
  std::cout << "G=" << mgxs.total.size() << std::endl;
  std::cout << "J=" << mgxs.total[0].size() << std::endl;
  MeshQuarterPincellCladded();
  Compare(mgxs);
}

}  // namespace

}  // namespace aether