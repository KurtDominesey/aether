#ifndef AETHER_EXAMPLES_COMPARE_TEST_H_
#define AETHER_EXAMPLES_COMPARE_TEST_H_

#include <hdf5.h>

#include <deal.II/base/hdf5.h>
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/eigen.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>

#include "example_test.h"

#include "base/petsc_block_block_wrapper.h"
#include "base/petsc_block_wrapper.h"
#include "base/stagnation_control.h"
#include "base/lapack_full_matrix.h"
#include "base/petsc_precondition_matrix.h"
#include "base/petsc_precondition_shell.h"
#include "base/slepc_transformation_preconditioner.h"
#include "sn/fixed_source_problem.h"
#include "sn/fission_problem.h"
#include "sn/fission_source.h"
#include "pgd/sn/inner_products.h"
#include "pgd/sn/fission_source_p.h"
#include "pgd/sn/energy_mg_fiss.h"
#include "pgd/sn/eigen_gs.h"
#include "pgd/sn/fixed_source_s_problem.h"
#include "pgd/sn/fission_s_problem.h"
#include "pgd/sn/shifted_s.h"
#include "pgd/sn/subspace_eigen.h"
#include "pgd/sn/subspace_jacobian_fd.h"
#include "pgd/sn/subspace_jacobian_pc.h"

template <int dim, int qdim>
class CompareTest : virtual public ExampleTest<dim, qdim> {
 protected:
  using ExampleTest<dim, qdim>::mesh;
  using ExampleTest<dim, qdim>::quadrature;
  using ExampleTest<dim, qdim>::dof_handler;
  using ExampleTest<dim, qdim>::mgxs;

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

  template <class TransportType>
  void RunFullOrder(dealii::BlockVector<double> &flux,
                    const dealii::BlockVector<double> &source,
                    const FixedSourceProblem<dim, qdim, TransportType> &problem,
                    const int max_iters, const double tol,
                    std::vector<double> *history_data = nullptr) {
    dealii::BlockVector<double> uncollided(source.get_block_indices());
    problem.sweep_source(uncollided, source);
    dealii::ReductionControl control_wg(500, tol*1e-2, 1e-2);
    dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg,
        dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData(32));
    FixedSourceGS<dealii::SolverGMRES<dealii::Vector<double>>, dim, qdim>
        preconditioner(problem.fixed_source, solver_wg);
    dealii::SolverControl control(max_iters, tol);
    control.enable_history_data();
    dealii::SolverRichardson<dealii::BlockVector<double>> solver(control);
    solver.connect([](const unsigned int iteration,
                      const double check_value,
                      const dealii::BlockVector<double>&) {
      std::cout << iteration << ": " << check_value << std::endl;
      return dealii::SolverControl::success;
    });
    try {
      solver.solve(problem.fixed_source, flux, uncollided, preconditioner);
    } catch (dealii::SolverControl::NoConvergence &failure) {
      failure.print_info(std::cout);
    }
    if (history_data != nullptr)
      *history_data = control.get_history_data();
  }

  template <class TransportType>
  void RunFullOrderCriticality(
      dealii::BlockVector<double> &flux,
      const dealii::BlockVector<double> &source,
      const FissionProblem<dim, qdim, TransportType> &problem,
      const int max_iters, const double tol,
      std::vector<double> *history_data = nullptr) {
    const bool use_slepc = true;
    dealii::SolverControl control(max_iters, tol);
    control.enable_history_data();
    if (use_slepc) {
      const int num_groups = mgxs->total.size();
      ::aether::PETScWrappers::BlockBlockWrapper fixed_source(
          num_groups, quadrature.size(), MPI_COMM_WORLD, 
          dof_handler.n_dofs(), dof_handler.n_dofs(), problem.fixed_source);
      ::aether::PETScWrappers::BlockBlockWrapper fission(
          num_groups, quadrature.size(), MPI_COMM_WORLD, 
          dof_handler.n_dofs(), dof_handler.n_dofs(), problem.fission);
      const int size = dof_handler.n_dofs() * quadrature.size() * num_groups;
      std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
      eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
      eigenvectors[0] = 1;
      // for (int i = 0; i < size; ++i)
      //   eigenvectors[0][i] = uncollided[i];
      eigenvectors[0] /= eigenvectors[0].l2_norm();
      std::vector<double> eigenvalues = {1.0};
      dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
      eigensolver.set_initial_space(eigenvectors);
      try {
        eigensolver.solve(fission, fixed_source, eigenvalues, eigenvectors);
      } catch (dealii::SolverControl::NoConvergence &failure) {
        failure.print_info(std::cout);
      }
      if (history_data != nullptr)
        *history_data = control.get_history_data();
      for (int i = 0; i < size; ++i)
        flux[i] = eigenvectors[0][i];
      std::cout << "EIGENVALUE: " << std::setprecision(10) 
                << eigenvalues[0] << std::endl;
    } else {
      // dealii::BlockVector<double> uncollided(source.get_block_indices());
      // problem.sweep_source(uncollided, source);
      dealii::ReductionControl control_wg(100, 1e-10, 1e-2);
      dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg);
      FixedSourceGS preconditioner(problem.fixed_source, solver_wg);
      // dealii::SolverControl control_fs(1, std::numeric_limits<double>::infinity);
      dealii::IterationNumberControl control_fs(1, 0);
      dealii::SolverRichardson<dealii::BlockVector<double>> solver_fs(control_fs);
      solver_fs.connect([](const unsigned int iteration,
                        const double check_value,
                        const dealii::BlockVector<double>&) {
        std::cout << iteration << ": " << check_value << std::endl;
        return dealii::SolverControl::success;
      });
      FissionSource fission_source(problem.fixed_source, problem.fission, 
                                   solver_fs, preconditioner);
      dealii::GrowingVectorMemory<dealii::BlockVector<double>> memory;
      dealii::EigenPower<dealii::BlockVector<double>> eigensolver(control, 
                                                                  memory);
      double k = 1.0;
      flux = 1;
      flux /= flux.l2_norm();
      eigensolver.solve(k, fission_source, flux);
      std::cout << "k-eigenvalue: " << std::setprecision(10) << k << std::endl;
    }
  }

  void RunPgd(pgd::sn::NonlinearGS &nonlinear_gs, const int num_modes,
              const int max_iters, const double tol, const bool do_update,
              std::vector<int> &unconverged, std::vector<double> &residuals) {
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
        nonlinear_gs.update();
      }
    }
  }

  double ComputeEigenvalue(
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

  void ComputeSvd(std::vector<dealii::BlockVector<double>> &svecs_spaceangle,
                  std::vector<dealii::Vector<double>> &svecs_energy,
                  const dealii::BlockVector<double> &flux,
                  const Transport<dim, qdim> &transport) {
    AssertDimension(svecs_spaceangle.size(), 0);
    AssertDimension(svecs_energy.size(), 0);
    const int num_groups = flux.n_blocks();
    const int num_qdofs = flux.block(0).size();
    std::cout << "initialize flux matrix\n";
    dealii::LAPACKFullMatrix_<double> flux_matrix(num_qdofs, num_groups);
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
  }

  void Compare(const int num_modes,
               const int max_iters_nonlinear,
               const double tol_nonlinear,
               const int max_iters_fullorder,
               const double tol_fullorder,
               const bool do_update,
               const bool precomputed_full=true,
               const bool precomputed_pgd=true,
               const bool do_eigenvalue=false) {
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
    const std::string filename_full = this->GetTestName() + "_full.h5";
    namespace HDF5 = dealii::HDF5;
    if (precomputed_full) {
      HDF5::File file(filename_full, HDF5::File::FileAccessMode::open);
      flux_full = file.open_dataset("flux_full").read<dealii::Vector<double>>();
    } else {
      std::vector<double> history_data;
      // run problem
      if (do_eigenvalue) {
        // RunFullOrderCriticality(flux_full, source_full, problem_full, 
        //     max_iters_fullorder, tol_fullorder, &history_data);
      } else {
        RunFullOrder(flux_full, source_full, problem_full, 
                    max_iters_fullorder, tol_fullorder, &history_data);
      }
      this->WriteFlux(flux_full, history_data, filename_full);
    }
    this->PlotFlux(flux_full, problem_full.d2m, mgxs->group_structure, "full");
    // Compute svd of full order
    std::cout << "compute svd\n";
    std::vector<dealii::BlockVector<double>> svecs_spaceangle;
    std::vector<dealii::Vector<double>> svecs_energy;
    ComputeSvd(svecs_spaceangle, svecs_energy, flux_full, 
               problem_full.transport);
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
    const std::string filename_pgd = this->GetTestName() + "_pgd.h5";
    if (precomputed_pgd) {
      // read from file
      HDF5::File file(filename_pgd, HDF5::File::FileAccessMode::open);
      for (int m = 0; m < num_modes; ++m) {
        const std::string mm = std::to_string(m);
        fixed_source_p.enrich(0);
        fixed_source_p.caches.back().mode = file.open_dataset(
            "modes_spaceangle"+mm).read<dealii::Vector<double>>();
        fixed_source_p.set_cache(fixed_source_p.caches.back());
        energy_mg.modes.push_back(file.open_dataset(
            "modes_energy"+mm).read<dealii::Vector<double>>());
      }
    } else {
      // run pgd
      std::vector<pgd::sn::LinearInterface*> linear_ops;
      std::unique_ptr<pgd::sn::NonlinearGS> nonlinear_gs;
      if (do_eigenvalue) {
        linear_ops = {&energy_fiss, &fission_source_p};
        nonlinear_gs =
            std::make_unique<pgd::sn::EigenGS>(linear_ops, num_materials, 1);
        auto eigen_gs = dynamic_cast<pgd::sn::EigenGS*>(nonlinear_gs.get());
        eigen_gs->initialize_guess();
      } else {
        linear_ops = {&energy_mg, &fixed_source_p};
        nonlinear_gs = std::make_unique<pgd::sn::NonlinearGS>(
            linear_ops, num_materials, 1, num_sources);
      }
      std::vector<int> unconverged;
      std::vector<double> residuals;
      std::cout << "run pgd\n";
      RunPgd(*nonlinear_gs, num_modes, max_iters_nonlinear, tol_nonlinear,
             do_update, unconverged, residuals);
      std::cout << "done running pgd\n";
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
        file.write_dataset(
            "modes_spaceangle"+mm, fixed_source_p.caches[m].mode.block(0));
        file.write_dataset("modes_energy"+mm, energy_mg.modes[m]);
      }
    }
    std::vector<dealii::BlockVector<double>> modes_spaceangle(num_modes,
        dealii::BlockVector<double>(quadrature.size(), dof_handler.n_dofs()));
    for (int m = 0; m < num_modes; ++m)
      modes_spaceangle[m] = fixed_source_p.caches[m].mode.block(0);
    if (true) {
      const int num_modes_s = num_modes;
      pgd::sn::FissionSProblem<dim, qdim> subspace_problem(
          dof_handler, quadrature, mgxs_one, boundary_conditions, num_modes_s);
      std::vector<std::vector<pgd::sn::InnerProducts>> inner_products_x( 
          num_modes_s, std::vector<pgd::sn::InnerProducts>(
            num_modes_s, pgd::sn::InnerProducts(num_materials, 1)));
      std::vector<std::vector<double>> inner_products_b(num_modes_s,
          std::vector<double>(num_sources));
      // normalize
      double norm = 0;
      for (int m = 0; m < num_modes; ++m)
        norm += std::pow(energy_mg.modes[m].l2_norm(), 2);
      norm = std::sqrt(norm);
      for (int m = 0; m < num_modes; ++m) {
        energy_mg.modes[m] /= norm;
        fixed_source_p.caches[m].mode *= norm;
      }
      for (int m = 0; m < num_modes; ++m) {
        // energy_mg.modes[m] = svecs_energy[m];
        // double norm = energy_mg.modes[m].l2_norm();
        // double norm = 0;
        // for (int g = 0; g < num_groups; ++g) {
        //   int g_rev = num_groups - 1 - g;
        //   double width = std::log(mgxs->group_structure[g+1]
        //                           / mgxs->group_structure[g]);
        //   norm += std::pow(energy_mg.modes[m][g_rev], 2) / width;
        // }
        // std::cout << std::sqrt(norm) << std::endl;
        // energy_mg.modes[m] /= std::sqrt(norm);
        // fixed_source_p.caches[m].mode.block(0) *= std::sqrt(norm);
      }
      for (int m = 0; m < num_modes_s; ++m) {
        energy_mg.get_inner_products(
            inner_products_x[m], inner_products_b[m], m, 0);
        // std::cout << "m " << m << "\n";
        for (int j = 0; j < num_materials; ++j) {
          double removal_d = 0;
          double removal_od = 0;
          for (int mp = 0; mp < num_modes_s; ++mp) {
            double removal = inner_products_x[m][mp].collision[j] -
                             inner_products_x[m][mp].scattering[j][0];
            if (m == mp)
              removal_d += removal;
            else
              removal_od += removal;
          }
          // std::cout << std::boolalpha << (removal_d >= removal_od) << ": "
          //           << removal_d << " >= " << removal_od << "\n";
        }
        // for (int mp = 0; mp < num_modes_s; ++mp) {
        //   std::cout << inner_products_x[m][mp].streaming << " ";
        // }
        // std::cout << inner_products_x[m][m].streaming << std::endl;
      }
      dealii::BlockVector<double> modes(
          num_modes_s, quadrature.size()*dof_handler.n_dofs());
      dealii::BlockVector<double> operated(modes);
      dealii::BlockVector<double> source(modes);
      dealii::BlockVector<double> ax(modes);
      dealii::BlockVector<double> bx(modes);
      dealii::Vector<double> scratch(source.block(0));
      dealii::Vector<double> dual(source.block(0));
      for (int m = 0; m < num_modes_s; ++m) {
        modes.block(m) = fixed_source_p.caches[m].mode.block(0);
        // modes.block(m) = m == 0 ? 1 : 0;
        for (int s = 0; s < num_sources; ++s) {
          scratch = sources_spaceangle[s];
          problem.transport.vmult_mass(dual, scratch);
          source.block(m).add(inner_products_b[m][s], dual);
          problem.transport.vmult_mass_inv(dual);
          for (int i = 0; i < scratch.size(); ++i) {
            AssertThrow(std::abs(scratch[i]-dual[i]) < 1e-12,
                        dealii::ExcInvalidState());
          }
        }
      }
      dealii::BlockVector<double> ax_(modes);
      dealii::BlockVector<double> bx_(modes);
      dealii::BlockVector<double> modes_copy(modes);
      subspace_problem.set_cross_sections(inner_products_x);
      subspace_problem.fission_s.vmult(ax_, modes);
      subspace_problem.fixed_source_s.vmult(bx_, modes);
      double rayleigh_ = (modes * ax_) / (modes * bx_);
      std::cout << "rayleigh_ " << rayleigh_ << "?\n";
      // subspace_problem.fission_s_gs.set_shift(rayleigh_);
      // subspace_problem.fission_s_gs.vmult(modes, modes_copy);
      // subspace_problem.fixed_source_s_gs.vmult(modes, ax_);
      double rayleigh = 1;
      /*
      subspace_problem.set_cross_sections(inner_products_x);
      subspace_problem.fixed_source_s.vmult(bx, modes);
      subspace_problem.fission_s.vmult(ax, modes);
      double rayleigh = (modes * ax) / (modes * bx);
      std::cout << "rayleigh quotient: " << rayleigh << std::endl;
      // subspace_problem.step(modes, inner_products_x);
      rayleigh = 1 / rayleigh;
      // rayleigh = 0.5;
      // rayleigh = 0;
      // subspace_problem.fission_s_gs.set_shift(rayleigh);
      */
      if (false) {
        dealii::IterationNumberControl control(10, 1e-8);
        dealii::SolverRichardson<dealii::BlockVector<double>> solver(control);
        // dealii::SolverGMRES<dealii::BlockVector<double>> solver(control/*,
        //     dealii::SolverGMRES<dealii::BlockVector<double>>::AdditionalData(
        //       30, false)*/);
        // solver.set_omega(1.2);
        solver.connect([](const unsigned int iteration,
                          const double check_value,
                          const dealii::BlockVector<double>&) {
          std::cout << iteration << ": " << check_value << std::endl;
          return dealii::SolverControl::success;
        });
        std::cout << "modes: " << modes.l2_norm() << std::endl;
        subspace_problem.fixed_source_s_gs.use_jacobi = false;
        if (false) {
          solver.solve(subspace_problem.fixed_source_s, modes, source, 
                      subspace_problem.fixed_source_s_gs
                      //  dealii::PreconditionIdentity()
                      );
          
        } else {
          for (int m = 0; m < num_modes; ++m) {
            // modes.block(m) = m == 0 ? 1 : 0;
          }
          subspace_problem.fission_s.vmult(source, modes);
          if (false) {
            aether::pgd::sn::ShiftedS shifted(subspace_problem.fission_s, 
                                              subspace_problem.fixed_source_s);
            shifted.shift = rayleigh;
            solver.solve(shifted, modes, source, 
                        subspace_problem.fission_s_gs
                        // dealii::PreconditionIdentity()
                        );
          } else {
            dealii::BlockVector<double> copy(modes);
            solver.solve(subspace_problem.fixed_source_s, modes, source,
                         subspace_problem.fixed_source_s_gs);
            modes.add(-rayleigh, copy);
          }
          subspace_problem.fixed_source_s.vmult(ax, modes);
          subspace_problem.fission_s.vmult(bx, modes);
          double rayleigh_new = (modes * ax) / (modes * bx);
          std::cout << "rayleigh quotient new: " << 1/rayleigh_new << std::endl;
        }
        std::cout << "modes: " << modes.l2_norm() << std::endl;
        subspace_problem.fixed_source_s.vmult(operated, modes);
        std::cout << "operated: " << operated.l2_norm() << std::endl;
        std::cout << "source: " << source.l2_norm() << std::endl;
        source -= operated;
        std::cout << "source: " << source.l2_norm() << std::endl;
      } else {
        /*
        std::cout << "ax, bx: " << ax.l2_norm() << ", " << bx.l2_norm() << "\n";
        ax.add(-rayleigh, bx);
        std::cout << "initial residual: " << ax.l2_norm() << std::endl;
        dealii::BlockVector<double> ax_mass_inv(ax);
        for (int m = 0; m < ax_mass_inv.n_blocks(); ++m)
          problem.transport.vmult_mass_inv(ax_mass_inv.block(m));
        std::cout << "initial residual: " << ax_mass_inv.l2_norm() << std::endl;
        bx = 0;
        subspace_problem.fixed_source_s_gs.vmult(bx, ax);
        std::cout << "fixed gs residual: " << bx.l2_norm() << std::endl;
        bx = 0;
        subspace_problem.fission_s_gs.vmult(bx, ax);
        std::cout << "fission gs residual: " << bx.l2_norm() << std::endl;
        const int num_qdofs = quadrature.size() * dof_handler.n_dofs();
        ::aether::PETScWrappers::BlockWrapper fixed_source_s(
            num_modes_s, MPI_COMM_WORLD, num_qdofs, num_qdofs,
            subspace_problem.fixed_source_s);
        ::aether::PETScWrappers::BlockWrapper fission_s_gs(
            num_modes_s, MPI_COMM_WORLD, num_qdofs, num_qdofs,
            subspace_problem.fission_s_gs);
        ::aether::PETScWrappers::BlockWrapper fission_s(
            num_modes_s, MPI_COMM_WORLD, num_qdofs, num_qdofs, 
            subspace_problem.fission_s);
        const int size = num_modes_s * num_qdofs;
        std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
        eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
        eigenvectors[0].compress(dealii::VectorOperation::insert);
        const double norm = modes.l2_norm();
        // for (int i = 0; i < size; ++i)
        //   eigenvectors[0][i] = modes[i] / norm;
        for (int i = 0; i < num_qdofs; ++i)
          eigenvectors[0][i] = 1;
        eigenvectors[0].compress(dealii::VectorOperation::insert);
        std::vector<double> eigenvalues = {0.999};
        dealii::SolverControl control(10, 1e-4);
        // dealii::SLEPcWrappers::SolverPower eigensolver(control);
        dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
        eigensolver.set_initial_space(eigenvectors);
        eigensolver.set_target_eigenvalue(rayleigh);
        // shift and invert
        using Shift 
            = dealii::SLEPcWrappers::TransformationShiftInvert::AdditionalData;
        dealii::SLEPcWrappers::TransformationShiftInvert shift_invert(
            MPI_COMM_WORLD, Shift(rayleigh));
        shift_invert.set_matrix_mode(ST_MATMODE_SHELL);
        dealii::IterationNumberControl control_inv(10, 1e-4);
        dealii::PETScWrappers::SolverGMRES solver_inv(control_inv, MPI_COMM_WORLD);
        // dealii::PETScWrappers::PreconditionNone preconditioner(fixed_source_s);
        // aether::PETScWrappers::PreconditionerMatrix preconditioner(fixed_source_s_gs);
        aether::PETScWrappers::PreconditionerShell preconditioner(fission_s_gs);
        // make sure preconditioner works
        dealii::PETScWrappers::MPI::Vector foo(eigenvectors[0]);
        dealii::PETScWrappers::MPI::Vector bar(foo);
        std::cout << "a: ";
        // preconditioner.vmult(bar, foo);
        solver_inv.initialize(preconditioner);
        // std::cout << "b: "; preconditioner.vmult(bar, foo);
        shift_invert.set_solver(solver_inv);
        // std::cout << "c: "; preconditioner.vmult(bar, foo);
        // eigensolver.set_transformation(shift_invert);
        // std::cout << "d: "; preconditioner.vmult(bar, foo);
        dealii::SolverControl control_dummy(1, 0);
        dealii::PETScWrappers::SolverPreOnly solver_pc(control_dummy);
        // dealii::IterationNumberControl control_pc(3, 1e-6);
        // dealii::PETScWrappers::SolverGMRES solver_pc(control_pc);
        solver_pc.initialize(preconditioner);
        aether::SLEPcWrappers::TransformationPreconditioner stprecond(
            MPI_COMM_WORLD, fission_s_gs);
        stprecond.set_matrix_mode(ST_MATMODE_SHELL);
        stprecond.set_solver(solver_pc);
        eigensolver.set_transformation(stprecond);
        try {
          std::cout << "to solve\n";
          eigensolver.solve(fission_s, fixed_source_s, eigenvalues, eigenvectors);
          for (int i = 0; i < eigenvectors[0].size(); ++i)
            modes[i] = eigenvectors[0][i];
        } catch (dealii::SolverControl::NoConvergence &failure) {
          failure.print_info(std::cout);
        } //catch (...) {}
        std::cout << "k-eigenvalue: " << 1/eigenvalues[0] << std::endl;
        // std::cout << "e: "; preconditioner.vmult(bar, foo);
        // dealii::PETScWrappers::MPI::Vector baz(foo);
        // fixed_source_s_gs.vmult(
        //     static_cast<dealii::PETScWrappers::VectorBase&>(baz), 
        //     static_cast<const dealii::PETScWrappers::VectorBase&>(foo));
        // for (int i = 0; i < size; ++i) {
        //   AssertThrow(bar[i] == baz[i], dealii::ExcInvalidState());
        // }
        */
        for (int m = 0; m < 2; ++m) {
          // fixed_source_p.get_inner_products(
          //   inner_products_x[m], inner_products_b[m], m, 0);
          for (int mp = 0; mp < 2; ++mp) {
            std::cout << m << "," << mp << "\n";
            std::cout << "stream " << inner_products_x[m][mp].streaming << "\n";
            for (int j = 0; j < num_materials; ++j) {
              std::cout << "matl " << j << ": ";
              std::cout << "collide " << inner_products_x[m][mp].collision[j]
                        << " scatter " << inner_products_x[m][mp].scattering[j][0]
                        << " fission " << inner_products_x[m][mp].fission[j]
                        << "\n";
            }
          }
        }
      }
      for (int m = 0; m < num_modes; ++m) {
        energy_fiss.modes.push_back(energy_mg.modes[m]);
        // double norm = modes.block(m).l2_norm();
        // modes.block(m) /= norm;
        // energy_fiss.modes[m] *= norm;
        // double norm = subspace_problem.transport.inner_product(
        //     modes.block(m), modes.block(m));
        // norm = std::sqrt(norm);
        // modes.block(m) /= norm;
        // energy_fiss.modes[m] *= norm;
      }
      // !!!
      // subspace_problem.fixed_source_s.get_inner_products_lhs(
      //     inner_products_x, modes);
      // energy_fiss.update(inner_products_x, 1e-7);
      // for (int m = 0; m < num_modes_s; ++m) {
      //   energy_fiss.get_inner_products(
      //       inner_products_x[m], inner_products_b[m], m, 0);
      // }
      // !!!
      double shift = rayleigh_;
      std::vector<std::vector<std::vector<aether::pgd::sn::InnerProducts>>>
          coefficients;
      // subspace_problem.fixed_source_s.get_inner_products_lhs(
      //     inner_products_x, modes);
      // shift = energy_fiss.update(inner_products_x);
      for (int i = 0; i < 0; ++i) {
        double tol = std::max(1e-7, std::pow(10, -6-i));
        std::cout << "nonlinear iteration: " << i << "\n";
        double residual = subspace_problem.step(modes, inner_products_x, /*shift,*/
                                                tol);
        // double norm = subspace_problem.l2_norm(modes);
        // double norm = modes.l2_norm();
        // modes /= norm;
        // for (int m = 0; m < num_modes_s; ++m) {
        //   double norm = subspace_problem.transport.inner_product(
        //       modes.block(m), modes.block(m));
        //   norm = std::sqrt(norm);
        //   modes.block(m) /= norm;
        //   energy_fiss.modes[m] *= norm;
        // }
        subspace_problem.fixed_source_s.get_inner_products_lhs(
            inner_products_x, modes);
        coefficients.push_back(inner_products_x);
        shift = energy_fiss.update(inner_products_x, /*residual*/ 
                                   tol);
        // for (int m = 0; m < num_modes_s; ++m) {
        //   double norm = energy_fiss.modes[m].l1_norm();
        //   modes.block(m) *= norm;
        //   energy_fiss.modes[m] /= norm;
        // }
        std::cout << shift << "\n";
        for (int m = 0; m < num_modes_s; ++m)
            energy_fiss.get_inner_products(
                inner_products_x[m], inner_products_b[m], m, 0);
        coefficients.push_back(inner_products_x);
      }
      // return;
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
      modes_all.block(modes_all.n_blocks()-1)[0] = shift;  // initial eigenvalue
      dealii::BlockVector<double> step(modes_all);
      aether::pgd::sn::SubspaceJacobianPC<dim, qdim> jacobian_pc(
          subspace_problem, energy_fiss, jacobian.inner_products_unperturbed,
          jacobian.k_eigenvalue);
      for (int i = 0; i < 15; ++i) {
        std::cout << "setting modes " << modes_all.l2_norm() << "\n";
        jacobian.set_modes(modes_all);
        jacobian_pc.modes = modes_all;
        // double k_energy = 
        //     energy_fiss.update(jacobian.inner_products_unperturbed[0], 1e-5);
        // std::cout << "k-energy: " << k_energy << "\n";
        // if (i == 0)
        //   modes_all.block(modes_all.n_blocks()-1)[0] = k_energy;
        for (int m = 0; m < num_modes_s; ++m)
          for (int g = 0; g < num_groups; ++g)
            energy_fiss.modes[m][g] = modes_all.block(1)[m*num_groups+g];
        std::cout << "set modes\n";
        // set initial guess
        step = modes_all;
        // step = 1;
        // step *= -1;
        // step[step.size()-1] *= 1e-2;
        step /= step.l2_norm();
        step *= jacobian.residual_unperturbed.l2_norm();
        // solve
        dealii::IterationNumberControl control(5, 1e-6);
        dealii::SolverFGMRES<dealii::BlockVector<double>> solver(control/*,
            dealii::SolverGMRES<dealii::BlockVector<double>>::AdditionalData(
              30, true, true, true)*/);
        solver.solve(jacobian, step, jacobian.residual_unperturbed, 
                     jacobian_pc);
        modes_all.add(1.0, step);
      }
      jacobian.set_modes(modes_all);
      jacobian_pc.modes = modes_all;
      return;
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
    GetL2ErrorsDiscrete(l2_errors_d, modes_spaceangle, energy_mg.modes, 
                        flux_full, problem.transport, table, "error_d");
    GetL2ErrorsMoments(l2_errors_m, modes_spaceangle, energy_mg.modes, 
                       flux_full, problem.transport, problem.d2m, table, 
                       "error_m");
    GetL2Norms(l2_norms, modes_spaceangle, energy_mg.modes, problem.transport,
               table, "norm");
    if (num_groups < 800) {
    std::cout << "residuals\n";
    GetL2Residuals(l2_residuals, fixed_source_p.caches, energy_mg.modes, 
                   source_full, problem.transport, problem.m2d, problem_full,
                   false, table, "residual");
    std::cout << "residuals streamed\n";
    GetL2Residuals(l2_residuals_streamed, fixed_source_p.caches, 
                   energy_mg.modes, source_full, problem.transport, problem.m2d,
                   problem_full, true, table, "residual_streamed");
    dealii::BlockVector<double> uncollided(source_full.get_block_indices());
    problem_full.sweep_source(uncollided, source_full);
    GetL2ResidualsFull(l2_residuals_swept, modes_spaceangle, energy_mg.modes, 
                       uncollided, problem.transport, problem_full, table, 
                       "residual_swept");
    }
    this->WriteConvergenceTable(table);
  }
};


#endif  // AETHER_EXAMPLES_COMPARE_TEST_H_