#include <deal.II/base/quadrature_lib.h>

#include "sn/discrete_to_moment.cpp"
#include "sn/moment_to_discrete.cpp"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

TEST(DiscreteMomentTest, IsotropicD2M) {
  const int qdim = 2;
  int num_ords_qdim = 4;
  int num_ords = std::pow(num_ords_qdim, qdim);
  dealii::QGauss<qdim> quadrature(num_ords_qdim);
  DiscreteToMoment<qdim> d2m(quadrature);
  int num_dofs = 16;
  dealii::BlockVector<double> discrete(num_ords, num_dofs);
  dealii::BlockVector<double> zeroth(1, num_dofs);
  discrete = dealii::numbers::PI;
  d2m.vmult(zeroth, discrete);
  for (int ord = 0; ord < num_ords; ++ord) {
    for (int i = 0; i < num_dofs; ++i)
      ASSERT_DOUBLE_EQ(zeroth.block(0)[i], discrete.block(ord)[i]);
  }
}

TEST(DiscreteMomentTest, IsotropicM2D) {
  const int qdim = 2;
  int num_ords_qdim = 4;
  int num_ords = std::pow(num_ords_qdim, qdim);
  dealii::QGauss<qdim> quadrature(num_ords_qdim);
  MomentToDiscrete<qdim> m2d(quadrature);
  int num_dofs = 16;
  dealii::BlockVector<double> zeroth(1, num_dofs);
  dealii::BlockVector<double> discrete(num_ords, num_dofs);
  zeroth = dealii::numbers::PI;
  m2d.vmult(discrete, zeroth);
  for (int ord = 0; ord < num_ords; ++ord)
    ASSERT_EQ(discrete.block(ord), zeroth.block(0));
}

}  // namespace

}  // namespace aether::sn