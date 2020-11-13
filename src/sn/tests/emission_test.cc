#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>

#include "sn/emission.h"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

TEST(EmissionTest, OneMaterial) {
  const int dim = 3;
  dealii::FE_DGQ<dim> fe(2);
  dealii::Triangulation<dim> mesh;
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 20, -1, 1);
  dealii::DoFHandler<dim> dof_handler(mesh);
  dof_handler.distribute_dofs(fe);
  int num_dofs = dof_handler.n_dofs();
  std::vector<std::vector<double>> chi = {{1.23}, {5.67}};
  Emission<dim> emission(dof_handler, chi);
  dealii::Vector<double> produced(num_dofs);
  dealii::BlockVector<double> emitted(chi.size(), num_dofs);
  double strength = 8.91;
  produced = strength;
  emission.vmult(emitted, produced);
  for (int g = 0; g < chi.size(); ++g) {
    for (int i = 0; i < num_dofs; ++i) {
      ASSERT_DOUBLE_EQ(emitted.block(g)[i], chi[g][0]*produced[i]);
    }
  }
}

}  // namespace

}  // namespace aether::sn