#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/convergence_table.h>

#include "pgd/sn/nonlinear_gs.cc"
#include "pgd/sn/fixed_source_p.h"
#include "pgd/sn/transport.h"
#include "pgd/sn/linear_interface.h"
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
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 64, 0, 3);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    int num_polar = 4;
    quadrature = dealii::QGauss<qdim>(num_polar);
    source.reinit(num_polar, dof_handler.n_dofs());
    flux.reinit(num_polar, dof_handler.n_dofs());
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_polar, fe.dofs_per_cell));
  }

  dealii::Triangulation<dim> mesh;
  dealii::Quadrature<qdim> quadrature;
  dealii::DoFHandler<dim> dof_handler;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> flux;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
};

TEST_F(FixedSourcePTest, NoCoupling) {
  boundary_conditions[0] = 1;
  boundary_conditions[1] = 0;
  pgd::sn::Transport<dim, qdim> transport(dof_handler, quadrature);
  Scattering scattering(dof_handler);
  std::vector<double> cross_sections_total = {1.0};
  std::vector<std::vector<double>> cross_sections_scatter = {{0.0}};
  pgd::sn::TransportBlock<dim, qdim> transport_block(
      transport, cross_sections_total, boundary_conditions);
  ScatteringBlock scattering_block(scattering, cross_sections_scatter[0]);
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
                                       cross_sections_total,
                                       cross_sections_scatter, sources);
  std::vector<pgd::sn::LinearInterface*> linear_ops = {&fixed_source_p};
  pgd::sn::NonlinearGS nonlinear_gs(linear_ops, 1, 1, 1);
  nonlinear_gs.enrich();
  // for (int i = 0; i < 20; ++i)
  nonlinear_gs.step(dealii::BlockVector<double>(),
                    dealii::BlockVector<double>());
  fixed_source_p.caches.back().streamed.print(std::cout);
  std::cout << nonlinear_gs.inner_products_x[0][0].streaming << std::endl;
  std::valarray<double> &collision = 
      nonlinear_gs.inner_products_x[0][0].collision;
  for (int mat = 0; mat < collision.size(); ++mat)
    std::cout << collision[mat] << ", ";
  std::cout << std::endl;
  // dealii::BlockVector<double> uncollided(source.get_block_indices());
  // transport_block.vmult(uncollided, source, false);
  // uncollided.print(std::cout);
  // fixed_source.vmult(flux, uncollided);
  // flux.print(std::cout);
}

}  // namespace

}  // namespace aether::pgd::sn