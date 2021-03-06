#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>

#include "base/petsc_block_vector.h"
#include "sn/scattering.h"
#include "sn/scattering_block.h"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

template <class BlockVectorType>
class ScatteringTest : public ::testing::Test {};

using BlockVectorTypes = ::testing::Types<
    dealii::BlockVector<double>, PETScWrappers::MPI::BlockVector >;
TYPED_TEST_CASE(ScatteringTest, BlockVectorTypes);

TYPED_TEST(ScatteringTest, OneMaterialIsotropic) {
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
  TypeParam source(1, num_dofs);
  TypeParam scattered(1, num_dofs);
  source = 1.2345;
  scattering_block.vmult(scattered, source);
  for (int i = 0; i < num_dofs; ++i) {
    ASSERT_DOUBLE_EQ(scattered.block(0)[i], cross_section*source.block(0)[i]);
  }
}

}  // namespace

}  // namespace aether::sn