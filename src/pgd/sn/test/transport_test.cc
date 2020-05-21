#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/convergence_table.h>

#include "pgd/sn/transport.h"
#include "sn/quadrature.h"
#include "sn/transport_block.h"
#include "functions/attenuated.h"
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"

namespace aether::pgd::sn {

namespace {

class PgdTransport1DTest : public ::testing::TestWithParam<int> {
 protected:
  static const int dim = 1;
  static const int qdim = 1;
  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 64, 0, 3);
    dealii::FE_DGQ<dim> fe(TestWithParam::GetParam());
    dof_handler.initialize(mesh, fe);
    int num_polar = 8;
    quadrature = dealii::QGauss<qdim>(num_polar);
    source.reinit(num_polar, dof_handler.n_dofs());
    flux.reinit(num_polar, dof_handler.n_dofs());
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_polar, fe.dofs_per_cell));
  }

  dealii::Triangulation<dim> mesh;
  aether::sn::QAngle<dim, qdim> quadrature;
  dealii::DoFHandler<dim> dof_handler;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> flux;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
};

TEST_P(PgdTransport1DTest, VoidSource) {
  double strength = 5.0;
  source = strength;
  std::vector<double> cross_sections = {0.0};
  double x0 = mesh.begin()->face(0)->vertex(0)(0);
  double x1 = mesh.last()->face(1)->vertex(0)(0);
  double length = x1 - x0;
  Transport<1> transport(dof_handler, quadrature);
  aether::sn::TransportBlock transport_block(transport, cross_sections, 
                                             boundary_conditions);
  transport_block.vmult(flux, source, false);
  dealii::BlockVector<double> flux_streamed(flux.get_block_indices());
  dealii::BlockVector<double> source_collided(flux.get_block_indices());
  transport.stream(flux_streamed, flux, boundary_conditions);
  transport.collide(source_collided, source);
  for (int n = 0; n < quadrature.size(); ++n) {
    double sum_flux_streamed = 0;
    double sum_source_collided = 0;
    for (int i = 0; i < dof_handler.n_dofs(); ++i) {
      EXPECT_NEAR(source_collided.block(n)[i], flux_streamed.block(n)[i], 1e-12);
      sum_flux_streamed += flux_streamed.block(n)[i];
      sum_source_collided += source_collided.block(n)[i];
    }
    EXPECT_NEAR(sum_flux_streamed, strength * length, 1e-12);
    EXPECT_NEAR(sum_source_collided, strength * length, 1e-12);
  }
}

TEST_P(PgdTransport1DTest, Attenuation) {
  double incident = 7.0;
  boundary_conditions[0] = incident;
  boundary_conditions[1] = incident;
  double cross_section = 5.0;
  std::vector<double> cross_sections = {cross_section};
  double x0 = mesh.begin()->face(0)->vertex(0)(0);
  double x1 = mesh.last()->face(1)->vertex(0)(0);
  double length = x1 - x0;
  int num_cycles = 2;
  double l2_error_streamed[num_cycles][quadrature.size()];
  double l2_error_collided[num_cycles][quadrature.size()];
  for (int cycle = 0; cycle < num_cycles; ++cycle) {
    if (cycle > 0) {
      mesh.refine_global();
      dof_handler.initialize(mesh, dof_handler.get_fe());
      source.reinit(quadrature.size(), dof_handler.n_dofs());
      flux.reinit(quadrature.size(), dof_handler.n_dofs());
    }
    Transport<1> transport(dof_handler, quadrature);
    aether::sn::TransportBlock transport_block(transport, cross_sections, 
                                              boundary_conditions);
    transport_block.vmult(flux, source, false);
    dealii::BlockVector<double> streamed(flux.get_block_indices());
    dealii::BlockVector<double> collided(flux.get_block_indices());
    transport.stream(streamed, flux, boundary_conditions);
    transport.collide(collided, flux);
    for (int n = 0; n < quadrature.size(); ++n) {
      double sum_streamed = 0;
      double sum_collided = 0;
      for (int i = 0; i < dof_handler.n_dofs(); ++i) {
        EXPECT_NEAR(streamed.block(n)[i] + cross_section * collided.block(n)[i], 
                    0, 1e-12);
        sum_streamed += streamed.block(n)[i];
        sum_collided += collided.block(n)[i];
      }
      sum_collided *= cross_section;
      double mu = std::abs(2 * quadrature.point(n)[0] - 1);
      double attenuation = std::exp(-cross_section * (length / mu));
      double delta = incident * (1 - attenuation);
      double integral_streamed = -mu * delta;
      double integral_collided = mu * delta;
      l2_error_streamed[cycle][n] = sum_streamed - integral_streamed;
      l2_error_collided[cycle][n] = sum_collided - integral_collided;
      if (cycle > 0) {
        double ratio_streamed = l2_error_streamed[cycle-1][n]
                                / l2_error_streamed[cycle][n];
        double ratio_collided = l2_error_collided[cycle-1][n]
                                / l2_error_streamed[cycle][n];
        double l2_conv_streamed = std::log(std::abs(ratio_streamed)) 
                                  / std::log(2.0);
        double l2_conv_collided = std::log(std::abs(ratio_collided))
                                  / std::log(2.0);
        // EXPECT_NEAR(l2_conv_streamed, 2*GetParam()+1, 1e-1);
        // EXPECT_NEAR(l2_conv_collided, 2*GetParam()+1, 1e-1);
      }
      EXPECT_NEAR(sum_streamed, integral_streamed, 
                  1e-4 * std::abs(integral_streamed));
      EXPECT_NEAR(sum_collided, integral_collided, 
                  1e-4 * std::abs(integral_collided));
    }
  }
}

INSTANTIATE_TEST_CASE_P(FEDegree, PgdTransport1DTest, ::testing::Range(0, 3));

}  // namespace

}  // namespace aether::pgd::sn