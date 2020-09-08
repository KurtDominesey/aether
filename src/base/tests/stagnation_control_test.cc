#include <deal.II/base/utilities.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <iostream>

#include "testmatrix.h"

#include "base/stagnation_control.h"
#include "gtest/gtest.h"

namespace aether {

namespace {

class StagnationControlTest : public ::testing::Test {
 protected:
  dealii::Vector<double> u;
  dealii::Vector<double> f;
  dealii::SparsityPattern structure;
  dealii::SparseMatrix<double> A;
  dealii::PreconditionJacobi<> preconditioner;

  void SetUp() override {
    const unsigned int size = 32;
    create_laplace_matrix(A, structure, size);
    f.reinit(A.m());
    u.reinit(A.m());
    f = 1.;
    preconditioner.initialize(A);
  }

  void check_solve(dealii::SolverControl &solver_control, 
                   const bool expected_result) {
    dealii::SolverCG<dealii::Vector<double>> solver(solver_control);
    u = 0.;
    f = 1.;
    bool success = false;
    try {
      solver.solve(A, u, f, preconditioner);
      std::cout << "Success. ";
      success = true;
    } catch (dealii::SolverControl::NoConvergence &e) {
      std::cout << "Failure. ";
    }
    std::cout << "Solver stopped after " << solver_control.last_step() 
              << " iterations" << std::endl;
    EXPECT_EQ(success, expected_result);
  }
};

TEST_F(StagnationControlTest, Fail) {
  StagnationControl control(20, 1e-3);
  check_solve(control, false);
}

}  // namespace

}  // namespace aether