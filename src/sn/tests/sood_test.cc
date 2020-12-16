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
#include <deal.II/lac/petsc_precondition.h>

#include "base/mgxs.h"
#include "base/petsc_block_block_wrapper.h"
#include "mesh/mesh.h"
#include "sn/quadrature_lib.h"
#include "sn/fission_problem.h"
#include "sn/fission_source.h"
#include "sn/fixed_source_gs.h"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

class SoodTest : public testing::Test {
 protected:
  static const int dim = 1;
  std::unique_ptr<Mgxs> mgxs;
  dealii::Triangulation<dim> mesh;
  dealii::DoFHandler<dim> dof_handler;
  QAngle<dim> quadrature;
  using BoundaryConditions = 
      std::vector<std::vector<dealii::BlockVector<double> > >;
  BoundaryConditions boundary_conditions;
  std::unique_ptr<FissionProblem<dim>> problem;

  void SetUp() override {
    // Pu-a cross-sections
    const int num_groups = 1;
    mgxs = std::make_unique<Mgxs>(num_groups, 1, 1);
    mgxs->nu_fission[0][0] = 3.24 * 0.081600;
    mgxs->scatter[0][0][0] = 0.225216;
    mgxs->total[0][0] = 0.32640;
    mgxs->chi[0][0] = 1;
    const int dim = 1;
    dealii::FE_DGQ<dim> fe(1);
    const double length = 1.853722;  // cm, critical slab half-length
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 50, -length, length);
    dof_handler.initialize(mesh, fe);
    quadrature = QPglc<dim>(128);
    // vacuum boundary conditions (for 1 group and 2 boundaries)
    boundary_conditions.resize(num_groups, 
        std::vector<dealii::BlockVector<double>>(dim == 1 ? 2 : 1,
          dealii::BlockVector<double>(quadrature.size(), fe.dofs_per_cell)));
    problem = std::make_unique<FissionProblem<dim>>(
        dof_handler, quadrature, *mgxs, boundary_conditions);
  }
};

TEST_F(SoodTest, PuAOneGroupIsotropicSlab) {
  const int num_groups = this->mgxs->total.size();
  dealii::ReductionControl control_wg(100, 1e-10, 1e-4);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver_wg(control_wg);
  FissionSource fission_source(
      this->problem->fixed_source, this->problem->fission, solver_wg, 
      dealii::PreconditionIdentity());
  dealii::SolverControl control(100, 1e-8);
  const int size = this->dof_handler.n_dofs() * this->quadrature.size();
  dealii::GrowingVectorMemory<dealii::BlockVector<double>> memory;
  dealii::EigenPower<dealii::BlockVector<double>> eigensolver(control, memory);
  double k = 0.5;  // bad initial guess
  dealii::BlockVector<double> flux(num_groups, size);
  flux = 1;
  eigensolver.solve(k, fission_source, flux);
  EXPECT_NEAR(k, 1, 1e-5);  // within one pcm of criticality
}

template <typename Solver>
class SoodSLEPcTest : public SoodTest {};

using Solvers = ::testing::Types<
    dealii::SLEPcWrappers::SolverPower,
    dealii::SLEPcWrappers::SolverArnoldi,
    dealii::SLEPcWrappers::SolverKrylovSchur,
    dealii::SLEPcWrappers::SolverGeneralizedDavidson,
    dealii::SLEPcWrappers::SolverJacobiDavidson
>;
TYPED_TEST_CASE(SoodSLEPcTest, Solvers);

TYPED_TEST(SoodSLEPcTest, PuAOneGroupIsotropicSlab) {
  const int num_groups = this->mgxs->total.size();
  dealii::ReductionControl control_wg(100, 1e-10, 1e-4);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver_wg(control_wg);
  PETScWrappers::FissionSource fission_source(
      this->problem->fixed_source, this->problem->fission, solver_wg, 
      dealii::PreconditionIdentity());
  dealii::SolverControl control(100, 1e-8);
  const int size = 
      this->dof_handler.n_dofs() * this->quadrature.size() * num_groups;
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
  std::vector<double> eigenvalues = {0.5};
  TypeParam eigensolver(control);
  eigensolver.solve(fission_source, eigenvalues, eigenvectors, 1);
  EXPECT_NEAR(eigenvalues[0], 1, 1e-5);
}

TYPED_TEST(SoodSLEPcTest, PuAOneGroupIsotropicSlabGeneralized) {
  const int num_groups = this->mgxs->total.size();
  ::aether::PETScWrappers::BlockBlockWrapper fixed_source(
      num_groups, this->quadrature.size(), MPI_COMM_WORLD, 
      this->dof_handler.n_dofs(), this->dof_handler.n_dofs(),
      this->problem->fixed_source);
  ::aether::PETScWrappers::BlockBlockWrapper fission(
      num_groups, this->quadrature.size(), MPI_COMM_WORLD, 
      this->dof_handler.n_dofs(), this->dof_handler.n_dofs(),
      this->problem->fission);
  dealii::SolverControl control(100, 1e-8);
  const int size = 
      this->dof_handler.n_dofs() * this->quadrature.size() * num_groups;
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
  eigenvectors[0] = 1;
  std::vector<double> eigenvalues = {0.5};
  TypeParam eigensolver(control);
  eigensolver.set_initial_space(eigenvectors);
  bool is_davidson = 
      dynamic_cast<dealii::SLEPcWrappers::SolverGeneralizedDavidson*>
      (&eigensolver) != nullptr ||
      dynamic_cast<dealii::SLEPcWrappers::SolverJacobiDavidson*>
      (&eigensolver) != nullptr;
  if (is_davidson) {
    eigensolver.solve(fission, fixed_source, eigenvalues, eigenvectors);
  } else {
    dealii::SLEPcWrappers::TransformationShiftInvert shift_invert(
        MPI_COMM_WORLD,
        dealii::SLEPcWrappers::TransformationShiftInvert::AdditionalData(0.9));
    shift_invert.set_matrix_mode(ST_MATMODE_SHELL);
    dealii::ReductionControl control_inv(100, 1e-10, 1e-4);
    dealii::PETScWrappers::SolverGMRES solver_inv(control_inv, MPI_COMM_WORLD);
    dealii::PETScWrappers::PreconditionNone preconditioner(fixed_source);
    solver_inv.initialize(preconditioner);
    shift_invert.set_solver(solver_inv);
    eigensolver.set_transformation(shift_invert);
    eigensolver.solve(fission, fixed_source, eigenvalues, eigenvectors);
  }
  EXPECT_NEAR(eigenvalues[0], 1, 1e-5);
}

}  // namespace

}  // namespace aether::sn