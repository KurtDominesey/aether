#include <deal.II/base/quadrature_lib.h>

#include "sn/discrete_to_moment.h"
#include "sn/moment_to_discrete.h"
#include "sn/quadrature_lib.h"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

template <typename T>
class DiscreteToMomentDimTest : public ::testing::Test {
 protected:
  static const int dim = T::value;
  void SetUp() override {
    if (dim == 1)
      quadrature = QPglc<dim>(4);
    else
      quadrature = QPglc<dim>(4, 4);
    d2m = std::make_unique<DiscreteToMoment<dim>>(quadrature);
  }
  QAngle<dim> quadrature;
  std::unique_ptr<DiscreteToMoment<dim>> d2m;
};

using Dims =
    ::testing::Types< std::integral_constant<int, 1>,
                      std::integral_constant<int, 2>,
                      std::integral_constant<int, 3> >;
TYPED_TEST_CASE(DiscreteToMomentDimTest, Dims);

TYPED_TEST(DiscreteToMomentDimTest, IsotropicD2M) {
  const int order = 1;
  const int num_dofs = 16;
  const int num_moments = this->d2m->n_block_rows(order);
  dealii::BlockVector<double> discrete(this->quadrature.size(), num_dofs);
  dealii::BlockVector<double> moments(num_moments, num_dofs);
  discrete = dealii::numbers::PI;
  moments = std::nan("a");
  this->d2m->vmult(moments, discrete);
  double tol = 1e-14;
  for (int i = 0; i < num_dofs; ++i) {
    // scalar flux (moment 0) should equal isotropic discrete flux
    for (int n = 0; n < this->quadrature.size(); ++n)
      EXPECT_NEAR(discrete.block(n)[i], moments.block(0)[i], tol);
    // anisotropic moments (>0) should be zero
    for (int lm = 1; lm < num_moments; ++lm)
      EXPECT_NEAR(0, moments.block(lm)[i], tol);
  }
}

TEST(DiscreteMomentTest, IsotropicD2M) {
  const int dim = 2;
  const int qdim = 2;
  int num_ords_qdim = 4;
  int num_ords = std::pow(num_ords_qdim, qdim);
  dealii::QGauss<qdim> q_gauss(num_ords_qdim);
  QAngle<dim, qdim> quadrature(q_gauss);
  DiscreteToMoment<dim, qdim> d2m(quadrature);
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
  const int dim = 2;
  const int qdim = 2;
  int num_ords_qdim = 4;
  int num_ords = std::pow(num_ords_qdim, qdim);
  dealii::QGauss<qdim> q_gauss(num_ords_qdim);
  QAngle<dim, qdim> quadrature(q_gauss);
  MomentToDiscrete<dim, qdim> m2d(quadrature);
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