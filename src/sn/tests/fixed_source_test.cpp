#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>

#include "../fixed_source.hpp"
#include "gtest/gtest.h"

namespace {

template <class SolverType>
class FixedSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 8, -1, 1);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    int num_ords_qdim = 4;
    int num_ords = std::pow(num_ords_qdim, qdim);
    quadrature = dealii::QGauss<qdim>(num_ords_qdim);
    int num_dofs = dof_handler.n_dofs();
  }

  static const int dim = 1;
  static const int qdim = 1;
  dealii::Triangulation<1> mesh;
  dealii::DoFHandler<dim> dof_handler;
  dealii::Quadrature<qdim> quadrature;
  std::vector<std::vector<dealii::BlockVector<double>>> boundary_conditions;
};

using SolverTypes =
    ::testing::Types< dealii::SolverRichardson<dealii::BlockVector<double>>,
                      dealii::SolverGMRES<dealii::BlockVector<double>>,
                      dealii::SolverFGMRES<dealii::BlockVector<double>>,
                      dealii::SolverBicgstab<dealii::BlockVector<double>> >;
TYPED_TEST_CASE(FixedSourceTest, SolverTypes);

TYPED_TEST(FixedSourceTest, IsotropicPureScattering) {
  const int dim = this->dim;
  const int qdim = this-> qdim;
  const int num_groups = 2;
  const int num_ords = this->quadrature.size();
  const int num_dofs = this->dof_handler.n_dofs();
  MomentToDiscrete<qdim> m2d(this->quadrature);
  DiscreteToMoment<qdim> d2m(this->quadrature);
  std::vector<WithinGroup<dim, qdim>> within_groups;
  std::vector<std::vector<ScatteringBlock<dim>>> downscattering(num_groups);
  std::vector<std::vector<ScatteringBlock<dim>>> upscattering(num_groups);
  std::vector<std::vector<double>> xs_total = {{1.0}, {1.0}};
  std::vector<std::vector<std::vector<double>>> xs_scatter = {
       {{0.0}, {1.0}},
       {{1.0}, {0.0}}
  };
  const int dofs_per_cell = this->dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::BlockVector<double>> boundary_conditions_g(
      2, dealii::BlockVector<double>(num_ords, dofs_per_cell));
  std::vector<std::vector<dealii::BlockVector<double>>> 
      boundary_conditions(num_groups, boundary_conditions_g);
  boundary_conditions[0][0] = 1;
  boundary_conditions[0][1] = 1;
  boundary_conditions[1][0] = 1;
  boundary_conditions[1][1] = 1;
  Scattering<dim> scattering(this->dof_handler);
  for (int g = 0; g < num_groups; ++g) {
    Transport<dim, qdim> transport(
        this->dof_handler, this->quadrature, xs_total[g], 
        boundary_conditions[g]);
    ScatteringBlock<dim> scattering_wg(scattering, xs_scatter[g][g]);
    // WithinGroup<dim, qdim> within_group(transport, m2d, scattering, d2m);
    within_groups.emplace_back(transport, m2d, scattering_wg, d2m);
    for (int up = g - 1; up >= 0; --up)
      downscattering[g].emplace_back(scattering, xs_scatter[g][up]);
    for (int down = g + 1; down < num_groups; ++down)
      upscattering[g].emplace_back(scattering, xs_scatter[g][down]);
  }
  FixedSource<dim, qdim> fixed_source(
      within_groups, downscattering, upscattering, m2d, d2m);
  dealii::BlockVector<double> source(num_groups, num_ords*num_dofs);
  dealii::BlockVector<double> uncollided(num_groups, num_ords*num_dofs);
  dealii::BlockVector<double> flux(num_groups, num_ords*num_dofs);
  // fixed_source.vmult(flux, source);
  dealii::SolverControl solver_control(2000, 1e-10);
  TypeParam solver(solver_control);
  for (int g = 0; g < within_groups.size(); ++g)
    within_groups[g].transport.vmult(uncollided.block(g), source.block(g),
                                     false);
  solver.solve(fixed_source, flux, uncollided, dealii::PreconditionIdentity());
  // flux.print(std::cout);
  std::cout << "iterations required " 
            << solver_control.last_step() 
            << std::endl;
  for (int g = 0; g < num_groups; ++g)
    for (int i = 0; i < flux.block(g).size(); ++i)
      ASSERT_NEAR(1, flux.block(g)[i], 1e-10);
}

}  // namespace