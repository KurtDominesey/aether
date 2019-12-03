#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>

#include "sn/quadrature.hpp"
#include "sn/transport.hpp"
#include "sn/transport_block.hpp"
#include "functions/attenuated.hpp"
#include "gtest/gtest.h"

namespace {

class Transport1DTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    const int dim = 1;
    const int qdim = 1;
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 64, x0, x1);
    dealii::FE_DGQ<dim> fe(TestWithParam::GetParam());
    dof_handler.initialize(mesh, fe);
    int num_ords_qdim = 8;
    int num_ords = std::pow(num_ords_qdim, qdim);
    quadrature = dealii::QGauss<qdim>(num_ords_qdim);
    quadrature = reorder(quadrature);
    int num_dofs = dof_handler.n_dofs();
    source.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(num_ords, fe.dofs_per_cell));
  }

  const double x0 = 0;
  const double x1 = 1;
  dealii::Triangulation<1> mesh;
  dealii::Quadrature<1> quadrature;
  dealii::DoFHandler<1> dof_handler;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> flux;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
};

TEST_P(Transport1DTest, Void) {
  std::vector<double> cross_sections = {0};
  int num_ords = quadrature.size();
  int num_dofs = dof_handler.n_dofs();
  for (int n = 0; n < num_ords; ++n)
    for (dealii::BlockVector<double> &boundary_condition : boundary_conditions)
      boundary_condition.block(n) = n + 1;
  Transport<1> transport(dof_handler, quadrature);
  TransportBlock<1> transport_block(transport, cross_sections,
                                    boundary_conditions);
  transport_block.vmult(flux, source, false);
  for (int n = 0; n < num_ords; ++n) {
    for (int i = 0; i < num_dofs; ++i) {
      ASSERT_NEAR(n + 1, flux.block(n)[i], 1e-10);
    }
  }
}

TEST_P(Transport1DTest, VoidReflected) {
  std::vector<double> cross_sections = {0};
  int num_ords = quadrature.size();
  int num_dofs = dof_handler.n_dofs();
  boundary_conditions.resize(1);
  for (int n = num_ords / 2; n < num_ords; ++n)
    boundary_conditions[0].block(n) = n;
  using Face = typename dealii::Triangulation<1>::face_iterator;
  for (int f = 0; f < dealii::GeometryInfo<1>::faces_per_cell; ++f) {
    Face face = mesh.last_active()->face(f);
    if (face->at_boundary())
      face->set_boundary_id(types::reflecting_boundary_id);
  }
  Transport<1> transport(dof_handler, quadrature);
  TransportBlock<1> transport_block(transport, cross_sections,
                                    boundary_conditions);
  transport_block.vmult(flux, source, false);
  for (int n = num_ords / 2; n < num_ords; ++n)
    for (int i = 0; i < num_dofs; ++i)
      ASSERT_NEAR(n, flux.block(n)[i], 1e-10);
  for (int n = 0; n < num_ords / 2; ++n)
    for (int i = 0; i < num_dofs; ++i)
      ASSERT_NEAR(n + num_ords / 2, flux.block(n)[i], 1e-10);
}

TEST_P(Transport1DTest, Attenuation) {
  int num_ords = quadrature.size();
  double incident = 1;
  std::vector<double> cross_sections = {1.0};
  std::vector<Attenuated> solutions;
  solutions.reserve(num_ords);
  for (int n = 0; n < num_ords; ++n)
    solutions.emplace_back(ordinate<1>(quadrature.point(n))[0],
                           cross_sections[0], incident, x0 , x1);
  dealii::ConvergenceTable convergence_table;
  int num_cycles = 2;
  std::vector<std::vector<double>> l2_errors(num_cycles,
                                             std::vector<double>(num_ords));
  int num_dofs = dof_handler.n_dofs();
  for (int cycle = 0; cycle < num_cycles; ++cycle) {
    if (cycle > 0) {
      mesh.refine_global();
      dof_handler.initialize(mesh, dof_handler.get_fe());
      int num_dofs = dof_handler.n_dofs();
      source.reinit(num_ords, num_dofs);
      flux.reinit(num_ords, num_dofs);
    }
    for (int n = 0; n < num_ords; ++n)
      for (dealii::BlockVector<double> &boundary_condition 
           : boundary_conditions) {
        boundary_condition.block(n).reinit(num_dofs);
        boundary_condition.block(n) = incident;
      }
    Transport<1> transport(dof_handler, quadrature);
    TransportBlock<1> transport_block(transport, cross_sections, 
                                      boundary_conditions);
    transport_block.vmult(flux, source, false);
    for (int n = 0; n < num_ords; ++n) {
      dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
      dealii::VectorTools::integrate_difference(
          dof_handler, flux.block(n), solutions[n], difference_per_cell,
          dealii::QGauss<1>(dof_handler.get_fe().degree + 2),
          dealii::VectorTools::L2_norm);
      double l2_error = difference_per_cell.l2_norm();
      l2_errors[cycle][n] = l2_error;
      if (cycle > 0) {
        double l2_conv =
            std::log(std::abs(l2_errors[cycle - 1][n] / l2_errors[cycle][n])) /
            std::log(2.);
        EXPECT_NEAR(l2_conv, GetParam()+1, 2e-2);
      }
      std::string key = "L2 " + std::to_string(n);
      convergence_table.add_value(key, l2_error);
      // dealii::Vector<double> solution_h(num_dofs);
      // dealii::VectorTools::interpolate(dof_handler, solutions[n], solution_h);
      if (cycle == num_cycles - 1)
        convergence_table.set_scientific(key, true);
    }
    flux = 0;
  }
  convergence_table.evaluate_all_convergence_rates(
    dealii::ConvergenceTable::RateMode::reduction_rate_log2);
  // convergence_table.write_text(std::cout);
}

template <int dim>
class Transported : public dealii::Function<dim> {
 using DoubleAt = std::function<double(const dealii::Point<dim>&)>;
 using TensorAt 
     = std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&)>;
 public:
  Transported(const dealii::Tensor<1, dim> ordinate,
              const double cross_section,
              const DoubleAt solution,
              const TensorAt grad)
      : dealii::Function<dim>(), ordinate(ordinate),
        cross_section(cross_section), solution(solution), grad(grad) {}
  double value(const dealii::Point<dim> &p, const unsigned int = 0) const {
    return ordinate * grad(p) + cross_section * solution(p);
  }
  
 protected:
  const dealii::Tensor<1, dim> ordinate;
  const double cross_section;
  const DoubleAt solution;
  const TensorAt grad;
};

TEST_P(Transport1DTest, ManufacturedCosine) {
  int num_ords = quadrature.size();
  double cross_section = 1.0;
  std::vector<double> cross_sections = {cross_section};
  using Solution = dealii::Functions::CosineFunction<1>;
  Solution solution;
  std::vector<Transported<1>> sources;
  sources.reserve(num_ords);
  for (int n = 0; n < num_ords; ++n)
    sources.emplace_back(
        ordinate<1>(quadrature.point(n)), cross_section,
        std::bind(&Solution::value, 
                  solution,
                  std::placeholders::_1, 
                  0),
        std::bind(&Solution::gradient, 
                  solution,
                  std::placeholders::_1,
                  0));
  dealii::ConvergenceTable convergence_table;
  int num_cycles = 2;
  std::vector<std::vector<double>> l2_errors(num_cycles,
                                             std::vector<double>(num_ords));
  for (int cycle = 0; cycle < num_cycles; ++cycle) {
    if (cycle > 0)
      mesh.refine_global();
    dof_handler.initialize(mesh, dof_handler.get_fe());
    int num_dofs = dof_handler.n_dofs();
    source.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    for (int n = 0; n < num_ords; ++n) {
      dealii::VectorTools::interpolate(dof_handler, sources[n], source.block(n));
      for (dealii::BlockVector<double> &boundary_condition 
           : boundary_conditions) {
        boundary_condition.block(n).reinit(num_dofs);
        boundary_conditions[0].block(n) = solution.value(dealii::Point<1>(x0));
        boundary_conditions[1].block(n) = solution.value(dealii::Point<1>(x1));
      }
    }
    // dealii::Vector<double> solution_h(num_dofs);
    // dealii::VectorTools::interpolate(dof_handler, solution, solution_h);
    Transport<1> transport(dof_handler, quadrature);
    TransportBlock<1> transport_block(transport, cross_sections,
                                      boundary_conditions);
    transport_block.vmult(flux, source, false);
    for (int n = 0; n < num_ords; ++n) {
      dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
      dealii::VectorTools::integrate_difference(
          dof_handler, flux.block(n), solution, difference_per_cell,
          dealii::QGauss<1>(dof_handler.get_fe().degree + 2),
          dealii::VectorTools::L2_norm);
      double l2_error = difference_per_cell.l2_norm();
      l2_errors[cycle][n] = l2_error;
      if (cycle > 0) {
        double l2_conv =
            std::log(std::abs(l2_errors[cycle - 1][n] / l2_errors[cycle][n])) /
            std::log(2.);
        EXPECT_NEAR(l2_conv, GetParam()+1, 2e-2);
      }
      std::string key = "L2 " + std::to_string(n);
      convergence_table.add_value(key, l2_error);
      if (cycle == num_cycles - 1)
        convergence_table.set_scientific(key, true);
    }
    flux = 0;
  }
  convergence_table.evaluate_all_convergence_rates(
    dealii::ConvergenceTable::RateMode::reduction_rate_log2);
  // convergence_table.write_text(std::cout);
}

INSTANTIATE_TEST_CASE_P(FEDegree, Transport1DTest, ::testing::Range(0, 4));

class Transport3DTest : public ::testing::Test {
 protected:
  static const int dim = 3;
  static const int qdim = 2;

  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 8, -1, 1);
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    int num_ords_qdim = 2;
    int num_ords = std::pow(num_ords_qdim, qdim) * qdim;
    dealii::QGauss<1> q_polar(num_ords_qdim);
    dealii::QIterated<1> q_azimuthal(q_polar, 4);
    quadrature = dealii::QAnisotropic<qdim>(q_polar, q_azimuthal);
    // AssertDimension(q_polar.size()*2, q_azimuthal.size());
    // AssertDimension(num_ords, quadrature.size());
    int num_dofs = dof_handler.n_dofs();
    num_ords = quadrature.size();
    source.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    boundary_conditions.resize(
        1, dealii::BlockVector<double>(num_ords, fe.dofs_per_cell));
  }

  dealii::Triangulation<dim> mesh;
  dealii::Quadrature<qdim> quadrature;
  dealii::DoFHandler<dim> dof_handler;
  dealii::BlockVector<double> source;
  dealii::BlockVector<double> flux;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
};

TEST_F(Transport3DTest, ManufacturedCosine) {
  int num_ords = quadrature.size();
  double cross_section = 0.5;
  std::vector<double> cross_sections = {cross_section};
  using Solution = dealii::Functions::CosineFunction<dim>;
  Solution solution;
  std::vector<Transported<dim>> sources;
  sources.reserve(num_ords);
  for (int n = 0; n < num_ords; ++n)
    sources.emplace_back(
        ordinate<dim>(quadrature.point(n)), cross_section,
        std::bind(&Solution::value, 
                  solution,
                  std::placeholders::_1, 
                  0),
        std::bind(&Solution::gradient, 
                  solution,
                  std::placeholders::_1,
                  0));
  dealii::ConvergenceTable convergence_table;
  int num_cycles = 2;
  std::vector<std::vector<double>> l2_errors(num_cycles,
                                             std::vector<double>(num_ords));
  for (int cycle = 0; cycle < num_cycles; ++cycle) {
    if (cycle > 0)
      mesh.refine_global();
    dof_handler.initialize(mesh, dof_handler.get_fe());
    int num_dofs = dof_handler.n_dofs();
    source.reinit(num_ords, num_dofs);
    flux.reinit(num_ords, num_dofs);
    for (int n = 0; n < num_ords; ++n)
      dealii::VectorTools::interpolate(dof_handler, sources[n], source.block(n));
    // dealii::Vector<double> solution_h(num_dofs);
    // dealii::VectorTools::interpolate(dof_handler, solution, solution_h);
    Transport<dim> transport(dof_handler, quadrature);
    TransportBlock<dim> transport_block(transport, cross_sections,
                                        boundary_conditions);
    transport_block.vmult(flux, source, false);
    for (int n = 0; n < num_ords; ++n) {
      dealii::Vector<double> difference_per_cell(mesh.n_active_cells());
      dealii::VectorTools::integrate_difference(
          dof_handler, flux.block(n), solution, difference_per_cell,
          dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
          dealii::VectorTools::L2_norm);
      double l2_error = difference_per_cell.l2_norm();
      l2_errors[cycle][n] = l2_error;
      if (cycle > 0) {
        double l2_conv =
            std::log(std::abs(l2_errors[cycle - 1][n] / l2_errors[cycle][n])) /
            std::log(2.);
        EXPECT_NEAR(l2_conv, dof_handler.get_fe().degree+1, 5e-2);
      }
      std::string key = "L2 " + std::to_string(n);
      convergence_table.add_value(key, l2_error);
      if (cycle == num_cycles - 1)
        convergence_table.set_scientific(key, true);
      // flux.print(std::cout);
    }
    flux = 0;
  }
  convergence_table.evaluate_all_convergence_rates(
    dealii::ConvergenceTable::RateMode::reduction_rate_log2);
  convergence_table.write_text(std::cout);
}

}  // namespace