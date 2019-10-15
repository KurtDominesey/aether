#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>

#include "../transport.hpp"
#include "gtest/gtest.h"

namespace {

TEST(TransportTest, Void) {
  const int dim = 1;
  const int qdim = 1;
  dealii::Triangulation<dim> mesh;
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 10, 0, 1);
  // std::map<char, std::vector<double> > cross_sections;
  // cross_sections['T'] = {0};
  std::vector<double> cross_sections = {0};
  dealii::DoFHandler<dim> dof_handler(mesh);
  dealii::FE_DGQ<dim> fe(0);
  dof_handler.distribute_dofs(fe);
  int num_ords_qdim = 8;
  int num_ords = std::pow(num_ords_qdim, qdim);
  dealii::QGauss<qdim> quadrature(num_ords);
  Transport<dim> transport(dof_handler, quadrature, cross_sections);
  int num_dofs = dof_handler.n_dofs();
  dealii::BlockVector<double> source(num_ords, num_dofs);
  dealii::BlockVector<double> flux(num_ords, num_dofs);
  for (int n = 0; n < num_ords; ++n) {
    flux.block(n) = 0;
    source.block(n) = 0;
  }
  transport.vmult(flux, source);
  for (int n = 0; n < num_ords; ++n) {
    for (int i = 0; i < num_dofs; ++i) {
      // std::cout << flux.block(n)[i];
      ASSERT_DOUBLE_EQ(1, flux.block(n)[i]);
    }
  }
}

}