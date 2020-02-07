#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/convergence_table.h>

#include "pgd/sn/nonlinear_gs.cc"
#include "pgd/sn/fixed_source_p.h"
#include "pgd/sn/transport.h"
#include "pgd/sn/linear_interface.h"
#include "pgd/sn/inner_products.h"
#include "sn/quadrature.h"
#include "sn/transport_block.h"
#include "sn/scattering.h"
#include "sn/scattering_block.h"
#include "functions/attenuated.h"
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"

namespace aether {

namespace {

using namespace aether::sn;

class FixedSourcePTest : public ::testing::Test {
 protected:
  static const int dim = 1;
  static const int qdim = 1;

  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 256, 0, length);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    int num_polar = 4;
    quadrature = dealii::QGauss<qdim>(num_polar);
    source.reinit(1, num_polar * dof_handler.n_dofs());
    flux.reinit(1, num_polar * dof_handler.n_dofs());
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_polar, fe.dofs_per_cell));
  }

  void Test(std::vector<double> &cross_sections_total_r,
            std::vector<std::vector<double>> &cross_sections_scatter_r,
            pgd::sn::InnerProducts &exact_x,
            std::vector<double> &exact_b) {
    pgd::sn::Transport<dim, qdim> transport(dof_handler, quadrature);
    Scattering scattering(dof_handler);
    std::vector<double> cross_sections_total_w = {std::nan("t")};
    std::vector<std::vector<double>> cross_sections_scatter_w = 
        {{std::nan("s")}};
    pgd::sn::TransportBlock<dim, qdim> transport_block(
        transport, cross_sections_total_w, boundary_conditions);
    ScatteringBlock scattering_block(scattering, cross_sections_scatter_w[0]);
    DiscreteToMoment d2m(quadrature);
    MomentToDiscrete m2d(quadrature);
    WithinGroup within_group(transport_block, m2d, scattering_block, d2m);
    std::vector<WithinGroup<dim>> within_groups = {within_group};
    std::vector<std::vector<ScatteringBlock<dim>>> downscatter = {{}};
    std::vector<std::vector<ScatteringBlock<dim>>> upscatter = {{}};
    std::vector<dealii::BlockVector<double>> sources = {source};
    FixedSource<dim, qdim> fixed_source(within_groups, downscatter, upscatter,
                                        m2d, d2m);
    pgd::sn::FixedSourceP fixed_source_p(fixed_source,
                                        cross_sections_total_w,
                                        cross_sections_scatter_w, 
                                        cross_sections_total_r,
                                        cross_sections_scatter_r,
                                        sources);
    std::vector<pgd::sn::LinearInterface*> linear_ops = {&fixed_source_p};
    pgd::sn::NonlinearGS nonlinear_gs(linear_ops, 1, 1, 1);
    nonlinear_gs.enrich();
    nonlinear_gs.step(dealii::BlockVector<double>(),
                      dealii::BlockVector<double>());
    pgd::sn::InnerProducts &numerical_x = nonlinear_gs.inner_products_x[0][0];
    std::vector<double> &numerical_b = nonlinear_gs.inner_products_b[0];
    TestInnerProductsX(numerical_x, exact_x);
    TestInnerProductsB(numerical_b, exact_b);
  }

  void TestInnerProductsX(pgd::sn::InnerProducts &numerical_x,
                          pgd::sn::InnerProducts &exact_x) {
    double tol = 1e-4;
    EXPECT_NEAR(numerical_x.streaming, exact_x.streaming, tol);
    ASSERT_EQ(numerical_x.collision.size(), exact_x.collision.size());
    for (int j = 0; j < numerical_x.collision.size(); ++j) {
      EXPECT_NEAR(numerical_x.collision[j], exact_x.collision[j], tol);
      ASSERT_EQ(numerical_x.scattering[j].size(), exact_x.scattering[j].size());
      for (int ell = 0; ell < numerical_x.scattering[j].size(); ++ell)
        EXPECT_NEAR(numerical_x.scattering[j][ell], exact_x.scattering[j][ell], 
                    tol);
    }
  }

  void TestInnerProductsB(std::vector<double> &numerical_b,
                          std::vector<double> &exact_b) {
    double tol = 1e-4;
    ASSERT_EQ(numerical_b.size(), exact_b.size());
    for (int i = 0; i < numerical_b.size(); ++i)
      EXPECT_NEAR(numerical_b[i], exact_b[i], tol);
  }

  const double length = 2;
  dealii::Triangulation<dim> mesh;
  dealii::Quadrature<qdim> quadrature;
  dealii::DoFHandler<dim> dof_handler;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> flux;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
};

TEST_F(FixedSourcePTest, NoCouplingAttenuation) {
  double strength = 5;
  double cross_section_total = 3;
  boundary_conditions[0] = strength;
  boundary_conditions[1] = strength;
  std::vector<double> cross_sections_total = {cross_section_total};
  std::vector<std::vector<double>> cross_sections_scatter = {{0}};
  pgd::sn::InnerProducts exact_x(1, 1);
  for (int n = 0; n < quadrature.size(); ++n) {
    double mu = std::abs(2 * quadrature.point(n)[0] - 1);
    double delta = std::exp(-2*cross_section_total*(length/mu)) - 1;
    double factor = std::pow(strength, 2) * (-mu / 2);
    exact_x.collision[0] += quadrature.weight(n) * factor * delta;
  }
  exact_x.streaming = -exact_x.collision[0];
  std::vector<double> exact_b = {0};
  Test(cross_sections_total, cross_sections_scatter, exact_x, exact_b);
}

TEST_F(FixedSourcePTest, NoCouplingUniform) {
  double strength = 5;
  double cross_section_total = 3;
  boundary_conditions[0] = strength;
  boundary_conditions[1] = strength;
  std::vector<double> cross_sections_total = {cross_section_total};
  std::vector<std::vector<double>> cross_sections_scatter = 
      {{cross_section_total*(1-1e-8)}};
  pgd::sn::InnerProducts exact_x(1, 1);
  exact_x.collision[0] = strength * cross_section_total * strength * length;
  exact_x.scattering[0][0] = exact_x.collision[0];
  std::vector<double> exact_b = {0};
  Test(cross_sections_total, cross_sections_scatter, exact_x, exact_b);
}

}  // namespace

}  // namespace aether::pgd::sn