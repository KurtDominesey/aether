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

class SoodTest : public testing::Test {
 protected:
  void Main() {
    // Pu-a cross-sections
    const int num_groups = 1;
    Mgxs mgxs(num_groups, 1, 1);
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
    const int size = dof_handler.n_dofs() * quadrature.size();
    this->Solve(fission_source, control_k, num_groups, size);
  }
  // this is ugly, but we can't override the Solve method if it's a template
  using FissionSourceInst = 
      FissionSource<1, 1, dealii::SolverGMRES<dealii::BlockVector<double>>, 
                          dealii::PreconditionIdentity>;
  virtual void Solve(FissionSourceInst &matrix, 
                     dealii::SolverControl &control, int num_groups, int size) {
    dealii::GrowingVectorMemory<dealii::BlockVector<double>> memory;
    dealii::EigenPower<dealii::BlockVector<double>> solver(control, memory);
    double k = 0.5;  // bad initial guess
    dealii::BlockVector<double> flux(num_groups, size);
    flux = 1;
    solver.solve(k, matrix, flux);
    EXPECT_NEAR(k, 1, 1e-5);  // within one pcm of criticality
  }
};

TEST_F(SoodTest, PuAOneGroupIsotropicSlab) {
  this->Main();
}

template <typename Solver>
class SoodSLEPcTest : public SoodTest {
 protected:
  void Solve(FissionSourceInst &matrix, dealii::SolverControl &control, 
             int num_groups, int size) override {
    const int size_v = num_groups * size;
    std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
    eigenvectors.emplace_back(MPI_COMM_WORLD, size_v, size_v);
    std::vector<double> eigenvalues = {0.5};
    Solver eigensolver(control);
    eigensolver.solve(matrix, eigenvalues, eigenvectors, 1);
    EXPECT_NEAR(eigenvalues[0], 1, 1e-5);
  }
};

using Solvers = ::testing::Types<
    dealii::SLEPcWrappers::SolverPower,
    dealii::SLEPcWrappers::SolverArnoldi,
    dealii::SLEPcWrappers::SolverKrylovSchur,
    dealii::SLEPcWrappers::SolverGeneralizedDavidson,
    dealii::SLEPcWrappers::SolverJacobiDavidson
>;
TYPED_TEST_CASE(SoodSLEPcTest, Solvers);

TYPED_TEST(SoodSLEPcTest, PuAOneGroupIsotropicSlab) {
  this->Main();
}

}  // namespace

}  // namespace aether::sn