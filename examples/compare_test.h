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
#include "base/slepc_solver.h"
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
#include "pgd/sn/shifted_s.h"
#include "pgd/sn/fission_source_shifted_s.h"

template <int dim, int qdim = dim == 1 ? 1 : 2>
class CompareTest : virtual public ExampleTest<dim, qdim> {
 protected:
  using ExampleTest<dim, qdim>::mesh;
  using ExampleTest<dim, qdim>::quadrature;
  using ExampleTest<dim, qdim>::dof_handler;
  using ExampleTest<dim, qdim>::mgxs;

  void WriteUniformFissionSource(
      std::vector<dealii::Vector<double>> &sources_energy,
      std::vector<dealii::BlockVector<double>> &sources_spaceangle);

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
  double RunFullOrderCriticality(
      dealii::BlockVector<double> &flux,
      const dealii::BlockVector<double> &source,
      const FissionProblem<dim, qdim, TransportType> &problem,
      const int max_iters, const double tol,
      std::vector<double> *history_data = nullptr) {
    const bool use_slepc = false; //flux.n_blocks() < 300;
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
      // FixedSourceGS preconditioner(problem.fixed_source, solver_wg);
      const int size = dof_handler.n_dofs() * quadrature.size() * num_groups;
      std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
      eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
      eigenvectors[0] = 1;
      // for (int i = 0; i < size; ++i)
      //   eigenvectors[0][i] = uncollided[i];
      eigenvectors[0] /= eigenvectors[0].l2_norm();
      std::vector<double> eigenvalues = {1.0};
      bool use_davidson = false;
      if (use_davidson) {
        dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
        eigensolver.set_initial_space(eigenvectors);
        try {
          eigensolver.solve(fission, fixed_source, eigenvalues, eigenvectors);
        } catch (dealii::SolverControl::NoConvergence &failure) {
          failure.print_info(std::cout);
        }
      } else {  // shift-and-invert Rayleigh
        aether::SLEPcWrappers::SolverRayleigh eigensolver(control);
        eigensolver.set_initial_space(eigenvectors);
        using ShiftInvert = dealii::SLEPcWrappers::TransformationShiftInvert;
        ShiftInvert shift_invert(
            MPI_COMM_WORLD, ShiftInvert::AdditionalData(eigenvalues[0]));
        shift_invert.set_matrix_mode(ST_MATMODE_SHELL);
        dealii::ReductionControl control_si(50, 1e-8, 1-2);
        dealii::PETScWrappers::SolverGMRES solver_si(control_si, MPI_COMM_WORLD);
        dealii::ReductionControl control_wg(250, 1e-8, 1e-2);
        dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg);
        FixedSourceGS fixed_source_gs(problem.fixed_source, solver_wg);
        aether::PETScWrappers::BlockBlockWrapper fixed_source_gs_petsc(
            num_groups, quadrature.size(), MPI_COMM_WORLD, 
            dof_handler.n_dofs(), dof_handler.n_dofs(), fixed_source_gs);
        aether::PETScWrappers::PreconditionerShell fixed_source_gs_pc(
            fixed_source_gs_petsc);
        solver_si.initialize(fixed_source_gs_pc);
        shift_invert.set_solver(solver_si);
        eigensolver.set_transformation(shift_invert);
        try {
          eigensolver.solve(fission, fixed_source, eigenvalues, eigenvectors);
        } catch (dealii::SolverControl::NoConvergence &failure) {
          failure.print_info(std::cout);
        }
      }
      if (history_data != nullptr)
        *history_data = control.get_history_data();
      for (int i = 0; i < size; ++i)
        flux[i] = eigenvectors[0][i];
      std::cout << "EIGENVALUE: " << std::setprecision(10) 
                << eigenvalues[0] << std::endl;
      return eigenvalues[0];
    } else {
      // dealii::BlockVector<double> uncollided(source.get_block_indices());
      // problem.sweep_source(uncollided, source);
      // dealii::ReductionControl control_wg(10, 1e-8, 1e-2);
      dealii::IterationNumberControl control_wg(10, 1e-10);
      // dealii::IterationNumberControl control_wg(50, 1e-10);
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
      fission_source.precondition_only = false;
      double k = 1.0;
      flux = 1;
      flux /= flux.l2_norm();
      bool use_rayleigh = false;
      if (use_rayleigh) {
        const int num_groups = mgxs->total.size();
        ::aether::PETScWrappers::BlockBlockWrapper fission_source_petsc(
            num_groups, quadrature.size(), MPI_COMM_WORLD, 
            dof_handler.n_dofs(), dof_handler.n_dofs(), fission_source);
        const int size = dof_handler.n_dofs() * quadrature.size() * num_groups;
        std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
        eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
        eigenvectors[0] = flux[0];
        std::vector<double> eigenvalues = {k};
        aether::SLEPcWrappers::SolverRayleigh eigensolver(control);
        eigensolver.set_initial_space(eigenvectors);
        using Shift = dealii::SLEPcWrappers::TransformationShift;
        Shift shift(MPI_COMM_WORLD, Shift::AdditionalData(k));
        shift.set_matrix_mode(ST_MATMODE_SHELL);
        dealii::PETScWrappers::SolverGMRES solver_shift(control_fs, MPI_COMM_WORLD);
        dealii::PETScWrappers::PreconditionNone pc_identity(fission_source_petsc);
        solver_shift.initialize(pc_identity);
        shift.set_solver(solver_shift);
        eigensolver.set_transformation(shift);
        try {
          eigensolver.solve(fission_source_petsc, eigenvalues, eigenvectors);
        } catch (dealii::SolverControl::NoConvergence &failure) {
          failure.print_info(std::cout);
        }
        if (history_data != nullptr)
          *history_data = control.get_history_data();
        for (int i = 0; i < size; ++i)
          flux[i] = eigenvectors[0][i];
        k = eigenvalues[0];
      } else {
        dealii::GrowingVectorMemory<dealii::BlockVector<double>> memory;
        using EigenPower = dealii::EigenPower<dealii::BlockVector<double>>;
        EigenPower eigensolver(control, memory);//,
                               //EigenPower::AdditionalData(1.1447201471513586));
        eigensolver.solve(k, fission_source, flux);
      }
      std::cout << "k-eigenvalue: " << std::setprecision(10) << k << std::endl;
      return k;
    }
  }

  void RunPgd(pgd::sn::NonlinearGS &nonlinear_gs, const int num_modes,
              const int max_iters, const double tol, const bool do_update,
              std::vector<int> &unconverged, std::vector<double> &residuals,
              std::vector<double> *eigenvalues);

  double ComputeEigenvalue(
        FixedSourceProblem<dim, qdim, pgd::sn::Transport<dim, qdim>> &problem,
        dealii::BlockVector<double> &flux, dealii::BlockVector<double> &source,
        Mgxs &mgxs_problem, double &denominator);

  void GetL2ErrorsDiscrete(
      std::vector<double> &l2_errors,
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const dealii::BlockVector<double> &reference,
      const pgd::sn::Transport<dim, qdim> &transport,
      dealii::ConvergenceTable &table,
      const std::string &key);

  void GetL2ErrorsMoments(
      std::vector<double> &l2_errors,
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const dealii::BlockVector<double> &reference,
      const pgd::sn::Transport<dim, qdim> &transport,
      const DiscreteToMoment<qdim> &d2m,
      dealii::ConvergenceTable &table,
      const std::string &key);

  void GetL2ErrorsFissionSource(
      std::vector<double> &l2_errors,
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const dealii::BlockVector<double> &reference,
      const pgd::sn::Transport<dim, qdim> &transport,
      const DiscreteToMoment<qdim> &d2m,
      const Production<dim> &production,
      dealii::ConvergenceTable &table,
      const std::string &key);

  void GetL2Norms(
      std::vector<double> &l2_norms,
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const pgd::sn::Transport<dim, qdim> &transport,
      dealii::ConvergenceTable &table,
      const std::string &key);

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
      const std::string &key);

  void GetL2ResidualsEigen(
      std::vector<double> &l2_residuals,
      const std::vector<pgd::sn::Cache> &caches,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const pgd::sn::Transport<dim, qdim> &transport,
      const MomentToDiscrete<qdim> &m2d,
      const FixedSourceProblem<dim, qdim> &problem,
      const std::vector<double> &eigenvalues,
      dealii::ConvergenceTable &table,
      const std::string &key);

  void GetL2ResidualsEigenMoments(
      std::vector<double> &l2_residuals,
      const std::vector<pgd::sn::Cache> &caches,
      const std::vector<dealii::Vector<double>> &modes_energy,
      const pgd::sn::Transport<dim, qdim> &transport,
      const DiscreteToMoment<qdim> &d2m,
      const FixedSourceProblem<dim, qdim> &problem,
      const std::vector<double> &eigenvalues,
      dealii::ConvergenceTable &table,
      const std::string &key);

  void GetL2ResidualsFull(
      std::vector<double> &l2_residuals,
      const std::vector<dealii::BlockVector<double>> &modes_spaceangle,
      const std::vector<dealii::Vector<double>> &modes_energy,
      dealii::BlockVector<double> &uncollided,
      const pgd::sn::Transport<dim, qdim> &transport,
      const FixedSourceProblem<dim, qdim> &problem,
      dealii::ConvergenceTable &table,
      const std::string &key);

  void ComputeSvd(std::vector<dealii::BlockVector<double>> &svecs_spaceangle,
                  std::vector<dealii::Vector<double>> &svecs_energy,
                  const dealii::BlockVector<double> &flux,
                  const Transport<dim, qdim> &transport,
                  const dealii::BlockVector<double> *flux_adjoint=nullptr,
                  std::vector<dealii::Vector<double>> *adjoints_energy=nullptr,
                  std::vector<dealii::BlockVector<double>>
                      *adjoints_spaceangle=nullptr);

  void Compare(int num_modes,
               const int max_iters_nonlinear,
               const double tol_nonlinear,
               const int max_iters_fullorder,
               const double tol_fullorder,
               const bool do_update,
               const bool do_minimax=false,
               const bool precomputed_full=true,
               const bool precomputed_pgd=true,
               const bool do_eigenvalue=false,
               const bool full_only=false,
               int num_modes_s=0,
               const bool guess_svd=false,
               const bool guess_spatioangular=false);
};


#endif  // AETHER_EXAMPLES_COMPARE_TEST_H_