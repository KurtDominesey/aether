#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>

#include "sn/quadrature.hpp"
#include "sn/within_group.hpp"
#include "functions/attenuated.hpp"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

template <class SolverType>
class WithinGroupTest : public ::testing::Test {
 protected:
  static const int dim = 1;
  static const int qdim = 1;

  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 128, -1, 1);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    int num_ords_qdim = 8;
    int num_ords = std::pow(num_ords_qdim, qdim);
    quadrature = dealii::QGauss<qdim>(num_ords_qdim);
    quadrature = reorder(quadrature);
    int num_dofs = dof_handler.n_dofs();
    source.reinit(num_ords, num_dofs);
    uncollided.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_ords, fe.dofs_per_cell));
  }

  dealii::Triangulation<dim> mesh;
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

TYPED_TEST(WithinGroupTest, IsotropicPureScatteringReflected) {
  const int dim = this->dim;
  const int qdim = this->qdim;
  const int num_ords = this->quadrature.size();
  const int num_dofs = this->dof_handler.n_dofs();
  using Face = typename dealii::Triangulation<dim>::face_iterator;
  for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
    Face face = this->mesh.last_active()->face(f);
    if (face->at_boundary())
      face->set_boundary_id(types::reflecting_boundary_id);
  }
  this->boundary_conditions.resize(1);
  this->boundary_conditions[0] = 1;
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
  within_group.transport.vmult(this->uncollided, this->source, false);
  dealii::SolverControl solver_control(300, 1e-10);
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

TYPED_TEST(WithinGroupTest, IsotropicInfiniteMedium) {
  const int dim = this->dim;
  const int qdim = this->qdim;
  const int num_ords = this->quadrature.size();
  const int num_dofs = this->dof_handler.n_dofs();
  using Cell = typename dealii::Triangulation<dim>::cell_iterator;
  using Face = typename dealii::Triangulation<dim>::face_iterator;
  for (Cell cell : {this->mesh.begin_active(), this->mesh.last_active()}) {
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
      Face face = cell->face(f);
      if (face->at_boundary())
        face->set_boundary_id(types::reflecting_boundary_id);
    }
  }
  this->boundary_conditions.resize(0);
  double cross_section_total = 1.0;
  double cross_section_scattering = 0.9;
  std::vector<double> cross_sections_total = {cross_section_total};
  std::vector<double> cross_sections_scattering = {cross_section_scattering};
  double strength = 1;
  this->source = strength;
  Transport<dim, qdim> transport(this->dof_handler, this->quadrature);
  TransportBlock<dim, qdim> transport_block(transport, cross_sections_total,
                                            this->boundary_conditions);
  Scattering<dim> scattering(this->dof_handler);
  ScatteringBlock<dim> scattering_block(scattering, cross_sections_scattering);
  MomentToDiscrete<qdim> m2d(this->quadrature);
  DiscreteToMoment<qdim> d2m(this->quadrature);
  WithinGroup<dim, qdim> within_group(
      transport_block, m2d, scattering_block, d2m);
  within_group.transport.vmult(this->uncollided, this->source, false);
  dealii::SolverControl solver_control(300, 1e-10);
  TypeParam solver(solver_control);
  solver.solve(within_group, this->flux, this->uncollided, 
               dealii::PreconditionIdentity());
  // std::cout << "iterations required " 
  //           << solver_control.last_step() 
  //           << std::endl;
  double solution = strength / (cross_section_total - cross_section_scattering);
  for (int n = 0; n < num_ords; ++n) {
    for (int i = 0; i < num_dofs; ++i) {
      ASSERT_NEAR(solution, this->flux.block(n)[i], 1e-10);
    }
  }
}

}  // namespace

}  // namespace aether::sn