#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>

#include "sn/production.h"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

TEST(ProductionTest, OneMaterial) {
  const int dim = 3;
  dealii::FE_DGQ<dim> fe(2);
  dealii::Triangulation<dim> mesh;
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 20, -1, 1);
  dealii::DoFHandler<dim> dof_handler(mesh);
  dof_handler.distribute_dofs(fe);
  int num_dofs = dof_handler.n_dofs();
  std::vector<std::vector<double>> nu_fission = {{1.23}, {5.67}};
  Production<dim> production(dof_handler, nu_fission);
  dealii::Vector<double> produced(num_dofs);
  dealii::BlockVector<double> flux_scalar(nu_fission.size(), num_dofs);
  double produced_total = 0;
  for (int g = 0; g < nu_fission.size(); ++g) {
    double strength = g + 1;
    produced_total += strength * nu_fission[g][0];
    flux_scalar.block(g) = strength;
  }
  production.vmult(produced, flux_scalar);
  for (int i = 0; i < num_dofs; ++i) {
    ASSERT_DOUBLE_EQ(produced[i], produced_total);
  }
}

}  // namespace

}  // namespace aether::sn