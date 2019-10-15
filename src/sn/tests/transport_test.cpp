#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>

#include "../transport.hpp"
#include "gtest/gtest.h"

namespace {

class TransportTest1D : public ::testing::Test {
 private:
  dealii::Triangulation<1> mesh;

 protected:
  void SetUp() override {
    const int dim = 1;
    const int qdim = 1;
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 10, 0, 1);
    dealii::FE_DGQ<dim> fe(3);
    dof_handler.initialize(mesh, fe);
    int num_ords_qdim = 8;
    int num_ords = std::pow(num_ords_qdim, qdim);
    quadrature = dealii::QGauss<qdim>(num_ords_qdim);
    int num_dofs = dof_handler.n_dofs();
    source.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_ords, fe.dofs_per_cell));
  }

  dealii::Quadrature<1> quadrature;
  dealii::DoFHandler<1> dof_handler;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> flux;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
};

TEST_F(TransportTest1D, Void) {
  std::vector<double> cross_sections = {0};
  int num_ords = quadrature.size();
  int num_dofs = dof_handler.n_dofs();
  for (int n = 0; n < num_ords; ++n)
    for (dealii::BlockVector<double> &boundary_condition : boundary_conditions)
      boundary_condition.block(n) = n;
  Transport<1> transport(dof_handler, quadrature, cross_sections,
                         boundary_conditions);
  transport.vmult(flux, source);
  for (int n = 0; n < num_ords; ++n) {
    for (int i = 0; i < num_dofs; ++i) {
      ASSERT_NEAR(n, flux.block(n)[i], 1e-10);
    }
  }
}

}