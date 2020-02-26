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
#include "functions/function_lib.h"
#include "sn/fixed_source_problem.cc"
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
  static const int num_mats = 2;
  const int num_groups = 7;
  const int num_ell = 1;
  const double pitch = 0.63;
  const double radius = 0.54;
  dealii::Triangulation<dim> mesh;

  Mgxs ReadMgxs() {
    const std::string filename = "/mnt/c/Users/kurt/Documents/projects/aether/examples/c5g7/c5g7.h5";
    const std::string temperature = "294K";
    const std::vector<std::string> materials =  {"water", "uo2"};
    AssertDimension(materials.size(), num_mats);
    Mgxs mgxs = read_mgxs(filename, temperature, materials);
    for (int g = 0; g < num_groups; ++g) {
      for (int j = 0; j < num_mats; ++j)
        mgxs.total[g][j] *= pitch / 2;
      for (int gp = 0; gp < num_groups; ++gp) {
        for (int j = 0; j < num_mats; ++j) {
          mgxs.scatter[g][gp][j] *= pitch / 2;
          std::cout << mgxs.scatter[g][gp][j] << " ";
        }
      }
      std::cout << std::endl;
    }
    return mgxs;
  }

  void MeshQuarterPincell() {
    mesh_quarter_pincell(mesh, {radius}, pitch, {1, 0});
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

  void MeshSquare() {
    dealii::GridGenerator::hyper_cube(mesh, -1, 1);
    mesh.refine_global(6);
  }

  void TransformMesh() {
    dealii::GridTools::scale(2 / pitch, mesh);
    dealii::Point<dim> center(0, 0);
    dealii::GridTools::shift(center, mesh);
    // Reset manifolds after transformation
    mesh.set_manifold(1,
        dealii::SphericalManifold<dim>(center));
    dealii::TransfiniteInterpolationManifold<dim> trans_manifold;
    trans_manifold.initialize(mesh);
    mesh.set_manifold(2, trans_manifold);
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
    mesh.refine_global(0);
  }

  void TestFullOrder(const Mgxs &mgxs) {
    // Create quadrature
    const QPglc<dim, qdim> quadrature(1, 1);
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
    using Solution = dealii::Functions::CosineFunction<dim>;
    Solution solution;
    dealii::Vector<double> solution_energy(num_groups);
    for (int g = 0; g < num_groups; ++g)
      solution_energy[g] = g + 1;
    // Run MMS refinement cycles
    dealii::ConvergenceTable convergence_table;
    int num_cycles = 2;
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
        AssertDimension(num_mats, 1);
        const int j = 0;
        double collision = solution_energy[g] * mgxs.total[g][j];
        double scattering = 0;
        for (int gp = 0; gp < num_groups; ++gp)
          scattering += solution_energy[gp] * mgxs.scatter[g][gp][j];
        for (int n = 0; n < quadrature.size(); ++n) {
          source_g.block(n).add(collision, source_iso);
          source_g.block(n).add(-scattering, source_iso);
        }
        source.block(g) = source_g;
      }
      // Run problem
      FixedSourceProblem<dim, qdim> problem(
          dof_handler, quadrature, mgxs, boundary_conditions);
      dealii::BlockVector<double> uncollided(source.get_block_indices());
      dealii::BlockVector<double> flux(source.get_block_indices());
      problem.sweep_source(uncollided, source);
      dealii::SolverControl control(500, 1e-8);
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
          double l2_error = difference_per_cell.l2_norm();
          l2_errors[cycle][g][n] = l2_error;
          if (cycle > 0) {
            double ratio = l2_errors[cycle-1][g][n] / l2_errors[cycle][g][n];
            double l2_conv = std::log(std::abs(ratio)) / std::log(2.0);
            EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree + 1, 5e-2);
          }
          std::string key = "g" + std::to_string(g) + "n" + std::to_string(n);
          convergence_table.add_value(key, l2_error);
          convergence_table.set_scientific(key, true);
        }
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
    convergence_table.write_text(std::cout);
  }

  void Test(const Mgxs &mgxs) {
    Mgxs mgxs_one(1, num_mats, num_ell);
    Mgxs mgxs_pseudo(1, num_mats, num_ell);
    for (int j = 0; j < num_mats; ++j) {
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
    using Solution = dealii::Functions::CosineFunction<dim>;
    const int num_solutions = 1;
    std::vector<Solution> solutions_spaceangle;
    std::vector<dealii::Vector<double>> solutions_energy(
        num_solutions, dealii::Vector<double>(num_groups));
    for (int i = 0; i < num_solutions; ++i) {
      for (int g = 0; g < num_groups; ++g)
        solutions_energy[i][g] = g + 1;
      solutions_spaceangle.emplace_back();
    }
    // Run MMS refinement cycles
    dealii::ConvergenceTable convergence_table;
    int num_cycles = 4;
    double l2_errors[num_cycles][quadrature.size()];
    double l2_errors_energy[num_cycles];
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
      if (cycle > 0) {
        mesh.refine_global();
        dof_handler.initialize(mesh, dof_handler.get_fe());
      }
      // Create sources
      const int num_sources = num_solutions + 2 * num_mats * num_solutions;
      std::vector<dealii::BlockVector<double>> sources_spaceangle(
          num_sources, dealii::BlockVector<double>(
              1, quadrature.size() * dof_handler.n_dofs()));
      std::vector<dealii::Vector<double>> sources_energy(
          num_sources, dealii::Vector<double>(num_groups));
      for (int i = 0; i < num_solutions; ++i) {
        dealii::BlockVector<double> source_spaceangle(quadrature.size(), 
                                                      dof_handler.n_dofs());
        auto grad = std::bind(&Solution::gradient, solutions_spaceangle[i], 
                              std::placeholders::_1, 0);
        for (int n = 0; n < quadrature.size(); ++n) {
          Streamed<dim> streamed(quadrature.ordinate(n), grad);
          dealii::VectorTools::interpolate(
              dof_handler, streamed, source_spaceangle.block(n));
        }
        sources_spaceangle[i].block(0) = source_spaceangle;
        // sources_spaceangle[i].block(0) *= 0;
        sources_energy[i] = solutions_energy[i];
        // collision and scattering
        dealii::Vector<double> source_spaceangle_iso(dof_handler.n_dofs());
        dealii::VectorTools::interpolate(
            dof_handler, solutions_spaceangle[i], source_spaceangle_iso);
        for (int j = 0; j < num_mats; ++j) {
          for (int g = 0; g < num_groups; ++g) {
            sources_energy[i+1+j][g] = mgxs.total[g][j] 
                                       * solutions_energy[i][g];
            for (int gp = 0; gp < num_groups; ++gp)
              sources_energy[i+1+num_mats+j][g] += mgxs.scatter[g][gp][j]
                                                   * solutions_energy[i][gp];
          }
          double collision = 0;
          double scattering = 0;
          for (int g = 0; g < num_groups; ++g) {
            collision += solutions_energy[i][g] * sources_energy[i+1+j][g];
            scattering += solutions_energy[i][g] 
                          * sources_energy[i+1+num_mats+j][g];
          }
          collision /= std::pow(solutions_energy[i].l2_norm(), 2);
          scattering /= std::pow(solutions_energy[i].l2_norm(), 2);
          std::cout << "COLLISION TRUE " << collision << std::endl;
          std::cout << "SCATTERING TRUE " << scattering << std::endl;
          // source_spaceangle.print(std::cout);
          dealii::Vector<double> source_spaceangle_iso_j(source_spaceangle_iso);
          std::vector<dealii::types::global_dof_index> dof_indices(
              fe.dofs_per_cell);
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
          sources_spaceangle[i+1+j].block(0) = source_spaceangle;
          sources_spaceangle[i+1+num_mats+j].block(0) = source_spaceangle;
          sources_spaceangle[i+1+num_mats+j].block(0) *= -1;
        }
      }
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
          linear_ops, num_mats, num_ell, num_sources);
      nonlinear_gs.enrich();
      for (int k = 0; k < 30; ++k) {
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
        std::string key = "L2 " + std::to_string(n);
        convergence_table.add_value(key, l2_error);
        convergence_table.set_scientific(key, true);
      }
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
    std::ofstream table_out("convergence.txt");
    convergence_table.write_text(table_out);
    convergence_table.write_tex(std::cout);
  }
};

TEST_F(C5G7EnergyTest, MmsPgd) {
  Mgxs mgxs = ReadMgxs();
  MeshQuarterPincell();
  Test(mgxs);
}

TEST_F(C5G7EnergyTest, MmsFull) {
  Mgxs mgxs = ReadMgxs();
  MeshQuarterPincell();
  TestFullOrder(mgxs);
}

}  // namespace

}  // namespace aether