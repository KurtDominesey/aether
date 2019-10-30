#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>

#include "../within_group.cpp"
#include "../../functions/attenuated.hpp"
#include "gtest/gtest.h"

namespace {

template <class SolverType>
class WithinGroupTest : public ::testing::Test {
 protected:
  static const int dim = 1;
  static const int qdim = 1;
 private:
  dealii::Triangulation<1> mesh;

 protected:
  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 128, -1, 1);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    int num_ords_qdim = 8;
    int num_ords = std::pow(num_ords_qdim, qdim);
    quadrature = dealii::QGauss<qdim>(num_ords_qdim);
    int num_dofs = dof_handler.n_dofs();
    source.reinit(num_ords, num_dofs);
    uncollided.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_ords, fe.dofs_per_cell));
  }

  dealii::DoFHandler<dim> dof_handler;
  dealii::Quadrature<qdim> quadrature;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> uncollided;
  dealii::BlockVector<double> flux;
};

using SolverTypes =
    ::testing::Types< dealii::SolverRichardson<dealii::BlockVector<double>>,
                      dealii::SolverGMRES<dealii::BlockVector<double>>,
                      dealii::SolverFGMRES<dealii::BlockVector<double>>,
                      dealii::SolverBicgstab<dealii::BlockVector<double>> >;
TYPED_TEST_CASE(WithinGroupTest, SolverTypes);

TYPED_TEST(WithinGroupTest, IsotropicPureScattering) {
  const int dim = this->dim;
  const int qdim = this->qdim;
  const int num_ords = this->quadrature.size();
  const int num_dofs = this->dof_handler.n_dofs();
  for (int n = 0; n < num_ords; ++n)
    for (dealii::BlockVector<double> &boundary_condition : 
         this->boundary_conditions)
      boundary_condition.block(n) = 1.0;
  std::vector<double> cross_sections_total = {1.0};
  std::vector<double> cross_sections_scattering = {1.0};
  Transport<dim, qdim> transport(this->dof_handler, this->quadrature);
  TransportBlock<dim, qdim> transport_block(transport, cross_sections_total,
                                            this->boundary_conditions);
  Scattering<dim> scattering(this->dof_handler);
  ScatteringBlock<dim> scattering_block(scattering, cross_sections_scattering);
  MomentToDiscrete<qdim> m2d(this->quadrature);
  DiscreteToMoment<qdim> d2m(this->quadrature);
  WithinGroup<dim, qdim> within_group(
      transport_block, m2d, scattering_block, d2m);
  this->source = 0;
  within_group.transport.vmult(this->uncollided, this->source, false);
  for (dealii::BlockVector<double> &boundary_condition : 
       this->boundary_conditions)
    for (int n = 0; n < num_ords; ++n)
      boundary_condition.block(n) = 0;
  this->flux = 0;
  dealii::SolverControl solver_control(200, 1e-10);
  TypeParam solver(solver_control);
  solver.solve(within_group, this->flux, this->uncollided, 
               dealii::PreconditionIdentity());
  // std::cout << "iterations required " 
  //           << solver_control.last_step() 
  //           << std::endl;
  for (int n = 0; n < num_ords; ++n) {
    for (int i = 0; i < num_dofs; ++i) {
      ASSERT_NEAR(1, this->flux.block(n)[i], 1e-10);
    }
  }
}

}  // namespace