#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>

#include "../within_group.cpp"
#include "../../functions/attenuated.hpp"
#include "gtest/gtest.h"

namespace {

class WithinGroupTest : public ::testing::Test {
 protected:
  static const int dim = 1;
  static const int qdim = 1;
 private:
  dealii::Triangulation<1> mesh;

 protected:
  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 5, -1, 1);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    int num_ords_qdim = 2;
    int num_ords = std::pow(num_ords_qdim, qdim);
    quadrature = dealii::QGauss<dim>(num_ords_qdim);
    int num_dofs = dof_handler.n_dofs();
    source.reinit(num_ords, num_dofs);
    uncollided.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_ords, fe.dofs_per_cell));
  }

  dealii::DoFHandler<1> dof_handler;
  dealii::Quadrature<1> quadrature;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> uncollided;
  dealii::BlockVector<double> flux;
};

TEST_F(WithinGroupTest, Void) {
  const int num_ords = quadrature.size();
  const int num_dofs = dof_handler.n_dofs();
  for (int n = 0; n < num_ords; ++n)
    for (dealii::BlockVector<double> &boundary_condition : boundary_conditions)
      boundary_condition.block(n) = 1;
  std::vector<double> cross_sections_total = {1.0};
  std::vector<double> cross_sections_scattering = {0.0};
  Transport<dim, qdim> transport(dof_handler, quadrature, cross_sections_total,
                                 boundary_conditions);
  Scattering<dim> scattering(dof_handler, cross_sections_scattering);
  MomentToDiscrete<qdim> m2d(quadrature);
  DiscreteToMoment<qdim> d2m(quadrature);
  WithinGroup<dim, qdim> within_group(transport, m2d, scattering, d2m);
  transport.vmult(uncollided, source);
  // for (int n = 0; n < num_ords; ++n) {
  //   for (int i = 0; i < num_dofs; ++i) {
  //     // ASSERT_NEAR(n + 1, flux.block(n)[i], 1e-10);
  //   }
  // }
  std::cout << "UNCOLLIDED\n";
  uncollided.print(std::cout);
  flux = 0;
  dealii::SolverControl solver_control(100, 1e-10);
  dealii::SolverRichardson<dealii::BlockVector<double>> solver(solver_control);
  solver.solve(within_group, flux, uncollided, dealii::PreconditionIdentity());
  std::cout << "FINAL FLUX\n";
  flux.print(std::cout);
  for (int n = 0; n < num_ords; ++n) {
    // uncollided.block(n).print(std::cout);
    // flux.block(n).print(std::cout);
    for (int i = 0; i < num_dofs; ++i) {
      // ASSERT_NEAR(1, flux.block(n)[i], 1e-10);
    }
  }
}

}