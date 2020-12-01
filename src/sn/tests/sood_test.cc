#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/eigen.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>

#include "base/mgxs.h"
#include "mesh/mesh.h"
#include "sn/quadrature_lib.h"
#include "sn/fission_problem.h"
#include "sn/fission_source.h"
#include "sn/fixed_source_gs.h"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

TEST(SoodTest, PuAOneGroupIsotropicSlab) {
  // Pu-a cross-sections
  Mgxs mgxs(1, 1, 1);
  mgxs.nu_fission[0][0] = 3.24 * 0.081600;
  mgxs.scatter[0][0][0] = 0.225216;
  mgxs.total[0][0] = 0.32640;
  mgxs.chi[0][0] = 1;
  const int dim = 1;
  dealii::FE_DGQ<dim> fe(1);
  dealii::Triangulation<dim> mesh;
  const double length = 1.853722;  // cm, critical slab half-length
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 50, -length, length);
  dealii::DoFHandler<dim> dof_handler(mesh);
  dof_handler.distribute_dofs(fe);
  const QPglc<dim> quadrature(128);
  // vacuum boundary conditions (for 1 group and 2 boundaries)
  const std::vector<std::vector<dealii::BlockVector<double>>>
      boundary_conditions(1, std::vector<dealii::BlockVector<double>>(2,
        dealii::BlockVector<double>(quadrature.size(), fe.dofs_per_cell)));
  FissionProblem<dim> problem(
      dof_handler, quadrature, mgxs, boundary_conditions);
  dealii::ReductionControl control(100, 1e-10, 1e-4);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver(control);
  FissionSource fission_source(problem.fixed_source, problem.fission, solver, 
                               dealii::PreconditionIdentity());
  dealii::SolverControl control_k(100, 1e-8);
  dealii::GrowingVectorMemory<dealii::BlockVector<double>> memory;
  dealii::EigenPower<dealii::BlockVector<double>> solver_k(control_k, memory);
  double k = 0.5;  // bad initial guess
  dealii::BlockVector<double> flux(1, quadrature.size()*dof_handler.n_dofs());
  flux = 1;
  solver_k.solve(k, fission_source, flux);
  EXPECT_NEAR(k, 1, 1e-5);  // within one pcm of criticality
  // works with SLEPc?
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, flux.size(), flux.size());
  std::vector<double> eigenvalues = {0.5};
  dealii::SLEPcWrappers::SolverKrylovSchur eigensolver(control);
  eigensolver.solve(fission_source, eigenvalues, eigenvectors, 1);
  EXPECT_NEAR(eigenvalues[0], 1, 1e-5);
}

}  // namespace

}  // namespace aether::sn