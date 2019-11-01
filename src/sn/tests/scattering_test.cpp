#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>

#include "sn/scattering.hpp"
#include "sn/scattering_block.hpp"
#include "gtest/gtest.h"

namespace {

TEST(ScatteringTest, OneMaterialIsotropic) {
  const int dim = 3;
  dealii::FE_DGQ<dim> fe(2);
  dealii::Triangulation<dim> mesh;
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 20, -1, 1);
  double cross_section = dealii::numbers::PI;
  std::vector<double> cross_sections = {cross_section};
  dealii::DoFHandler<dim> dof_handler(mesh);
  dof_handler.distribute_dofs(fe);
  int num_dofs = dof_handler.n_dofs();
  Scattering<dim> scattering(dof_handler);
  ScatteringBlock<dim> scattering_block(scattering, cross_sections);
  dealii::BlockVector<double> source(1, num_dofs);
  dealii::BlockVector<double> scattered(1, num_dofs);
  scattering_block.vmult(scattered, source);
  for (int i = 0; i < num_dofs; ++i) {
    ASSERT_DOUBLE_EQ(scattered.block(0)[i], cross_section*source.block(0)[i]);
  }
}

}  // namespace