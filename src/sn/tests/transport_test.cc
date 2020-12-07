#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>

#include "base/petsc_block_vector.h"
#include "sn/quadrature.h"
#include "sn/quadrature_lib.h"
#include "sn/transport.h"
#include "sn/transport_block.h"
#include "functions/attenuated.h"
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"

namespace aether::sn {

namespace {

class Transport1DTest : public ::testing::TestWithParam<int> {
 protected:
  static const int dim = 1;
  static const int qdim = 1;
  void SetUp() override {
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 64, 0, 1);
    dealii::FE_DGQ<dim> fe(TestWithParam::GetParam());
    dof_handler.initialize(mesh, fe);
    int num_polar = 4;
    quadrature = QPglc<dim, qdim>(num_polar);
    source.reinit(quadrature.size(), dof_handler.n_dofs());
    flux.reinit(quadrature.size(), dof_handler.n_dofs());
    boundary_conditions.resize(
        2, dealii::BlockVector<double>(quadrature.size(), fe.dofs_per_cell));
  }

  dealii::Triangulation<dim> mesh;
  QPglc<dim, qdim> quadrature;
  dealii::DoFHandler<dim> dof_handler;
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
      ASSERT_NEAR(num_ords-1 - n, flux.block(n)[i], 1e-10);
}

TEST_P(Transport1DTest, Attenuation) {
  int num_ords = quadrature.size();
  double incident = 1;
  std::vector<double> cross_sections = {1.0};
  std::vector<Attenuated> solutions;
  double x0 = mesh.begin()->face(0)->vertex(0)(0);
  double x1 = mesh.last()->face(1)->vertex(0)(0);
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
      if (cycle == num_cycles - 1)
        convergence_table.set_scientific(key, true);
    }
  }
  convergence_table.evaluate_all_convergence_rates(
    dealii::ConvergenceTable::RateMode::reduction_rate_log2);
}

INSTANTIATE_TEST_CASE_P(FEDegree, Transport1DTest, ::testing::Range(0, 4));

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

template <typename T>
class TransportMmsTest : public ::testing::Test {
 protected:
  using BlockVectorType = typename T::second_type;
  static const int dim = T::first_type::value;
  static const int qdim = dim == 1 ? 1 : 2;
  void SetUp() override {
    dealii::FE_DGQ<dim> fe(1);
    dof_handler.set_fe(fe);
    int num_polar = 2;
    if (qdim == 1) {
      quadrature = QPglc<dim, qdim>(num_polar);
    } else {
      AssertDimension(qdim, 2);
      int num_azimuthal = num_polar;
      quadrature = QPglc<dim, qdim>(num_polar, num_azimuthal);
    }
    boundary_conditions.resize(
        dim == 1 ? 2 : 1,
        dealii::BlockVector<double>(quadrature.size(), fe.dofs_per_cell));
    cross_section = 0.5;
  }
  template <typename Solution>
  void Test(Solution &solution, int num_sweeps = 1) {
    int num_ords = quadrature.size();
    std::vector<double> cross_sections = {cross_section};
    std::vector<Transported<dim>> sources;
    sources.reserve(num_ords);
    for (int n = 0; n < num_ords; ++n)
      sources.emplace_back(
          ordinate<dim>(this->quadrature.point(n)), this->cross_section,
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
        dealii::VectorTools::interpolate(dof_handler, sources[n], 
                                         source.block(n));
      Transport<dim> transport(dof_handler, quadrature);
      TransportBlock<dim> transport_block(transport, cross_sections,
                                          boundary_conditions);
      for (int sweep = 0; sweep < num_sweeps; ++sweep)
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
              std::log(std::abs(l2_errors[cycle - 1][n] / l2_errors[cycle][n])) 
              / std::log(2.);
          EXPECT_NEAR(l2_conv, this->dof_handler.get_fe().degree+1, 5e-2);
        }
        std::string key = "L2 " + std::to_string(n);
        convergence_table.add_value(key, l2_error);
        if (cycle == num_cycles - 1)
          convergence_table.set_scientific(key, true);
      }
    }
    convergence_table.evaluate_all_convergence_rates(
      dealii::ConvergenceTable::RateMode::reduction_rate_log2);
    convergence_table.write_text(std::cout);
  }

  dealii::Triangulation<dim> mesh;
  QPglc<dim, qdim> quadrature;
  dealii::DoFHandler<dim> dof_handler;
  BlockVectorType source;
  BlockVectorType flux;
  std::vector<dealii::BlockVector<double>> boundary_conditions;
  double cross_section;
};

using Dimensions = ::testing::Types<
    std::pair< std::integral_constant<int, 1>, dealii::BlockVector<double> >,
    std::pair< std::integral_constant<int, 2>, dealii::BlockVector<double> >,
    std::pair< std::integral_constant<int, 3>, dealii::BlockVector<double> >,
    std::pair< std::integral_constant<int, 1>, PETScWrappers::MPI::BlockVector >,
    std::pair< std::integral_constant<int, 2>, PETScWrappers::MPI::BlockVector >,
    std::pair< std::integral_constant<int, 3>, PETScWrappers::MPI::BlockVector > >;
TYPED_TEST_CASE(TransportMmsTest, Dimensions);

TYPED_TEST(TransportMmsTest, VacuumCosine) {
  dealii::GridGenerator::subdivided_hyper_cube(this->mesh, 8, -1, 1);
  dealii::Functions::CosineFunction<this->dim> solution;
  this->Test(solution);
}

TYPED_TEST(TransportMmsTest, ReflectedInlineCosine) {
  static const int dim = this->dim;
  int num_cells = 18;
  std::vector<unsigned int> num_cells_by_dim(dim, num_cells);
  num_cells_by_dim.back() /= 2;
  if (dim == 1)
    dealii::GridGenerator::subdivided_hyper_rectangle(
        this->mesh, num_cells_by_dim, 
        dealii::Point<dim>(-1), dealii::Point<dim>(0), true);
  else if (dim == 2)
    dealii::GridGenerator::subdivided_hyper_rectangle(
        this->mesh, num_cells_by_dim,
        dealii::Point<dim>(-1, -1), dealii::Point<dim>(1, 0), true);
  else if (dim == 3)
    dealii::GridGenerator::subdivided_hyper_rectangle<dim>(
        this->mesh, num_cells_by_dim, 
        dealii::Point<dim>(-1, -1, -1), dealii::Point<dim>(1, 1, 0), true);
  else
    throw dealii::ExcInvalidState();
  int boundary_id_top = 2 * dim - 1;
  using Cell = typename dealii::Triangulation<dim>::cell_iterator;
  using Face = typename dealii::Triangulation<dim>::face_iterator;
  for (Cell cell = this->mesh.begin(); cell != this->mesh.end(); cell++) {
    cell->set_material_id(0);
    if (!cell->at_boundary())
      continue;
    for (int f = 0; f < dealii::GeometryInfo<this->dim>::faces_per_cell; ++f) {
      Face face = cell->face(f);
      if (!face->at_boundary())
        continue;
      if (face->boundary_id() == boundary_id_top)
        face->set_boundary_id(types::reflecting_boundary_id);
      else
        face->set_boundary_id(0);
    }
  }
  this->boundary_conditions.resize(1);
  dealii::Functions::CosineFunction<dim> solution;
  this->Test(solution);
}

TYPED_TEST(TransportMmsTest, ReflectedOnceCosine) {
  static const int dim = this->dim;
  int num_cells = 18;
  std::vector<unsigned int> num_cells_by_dim(dim, num_cells);
  num_cells_by_dim.back() /= 2;
  if (dim == 1)
    dealii::GridGenerator::subdivided_hyper_rectangle(
        this->mesh, num_cells_by_dim, 
        dealii::Point<dim>(0), dealii::Point<dim>(1), true);
  else if (dim == 2)
    dealii::GridGenerator::subdivided_hyper_rectangle(
        this->mesh, num_cells_by_dim,
        dealii::Point<dim>(-1, 0), dealii::Point<dim>(1, 1), true);
  else if (dim == 3)
    dealii::GridGenerator::subdivided_hyper_rectangle<dim>(
        this->mesh, num_cells_by_dim, 
        dealii::Point<dim>(-1, -1, 0), dealii::Point<dim>(1, 1, 1), true);
  else
    throw dealii::ExcInvalidState();
  int boundary_id_bottom = 2 * dim - 2;
  using Cell = typename dealii::Triangulation<dim>::cell_iterator;
  using Face = typename dealii::Triangulation<dim>::face_iterator;
  for (Cell cell = this->mesh.begin(); cell != this->mesh.end(); cell++) {
    cell->set_material_id(0);
    if (!cell->at_boundary())
      continue;
    for (int f = 0; f < dealii::GeometryInfo<this->dim>::faces_per_cell; ++f) {
      Face face = cell->face(f);
      if (!face->at_boundary())
        continue;
      if (face->boundary_id() == boundary_id_bottom)
        face->set_boundary_id(types::reflecting_boundary_id);
      else
        face->set_boundary_id(0);
    }
  }
  this->boundary_conditions.resize(1);
  dealii::Functions::CosineFunction<dim> solution;
  dealii::Triangulation<dim> mesh_coarse;
  mesh_coarse.copy_triangulation(this->mesh);
  // sweep once: this should fail for half the ordinates
  int num_failures = this->quadrature.size() / 2;
  std::string msg_failure =
      "The difference between l2_conv and this->dof_handler.get_fe().degree+1";
  if (num_failures == 1)
    EXPECT_NONFATAL_FAILURE(this->Test(solution, 1), msg_failure);
  else
    EXPECT_NONFATAL_FAILURE(
        // this expectation will fail because we encounter more than one failure
        EXPECT_NONFATAL_FAILURE(this->Test(solution, 1), msg_failure),
        "Expected: 1 non-fatal failure\n"
        "  Actual: "+std::to_string(num_failures)+" failures");
  // sweep twice (using original mesh)
  this->mesh = std::move(mesh_coarse);
  this->dof_handler.initialize(this->mesh, this->dof_handler.get_fe());
  this->Test(solution, 2);
}

using OneD = 
    std::pair<std::integral_constant<int, 1>, dealii::BlockVector<double> >;
class TransportMms1DTest
    : public TransportMmsTest<OneD>,
      public ::testing::WithParamInterface<int> {
 protected:
  void SetUp() override {
    TransportMmsTest::SetUp();
    // parameterize the degree of the finite elements
    dealii::FE_DGQ<dim> fe(WithParamInterface::GetParam());
    dof_handler.set_fe(fe);
    for (dealii::BlockVector<double> &boundary_condition : boundary_conditions)
      for (int n = 0; n < boundary_condition.n_blocks(); ++n)
        boundary_condition.block(n).reinit(fe.dofs_per_cell);
  }
  template <typename Solution>
  void Test(Solution &solution) {
    dealii::Point<1> &x_first = mesh.begin_active()->face(0)->vertex(0);
    dealii::Point<1> &x_last = mesh.last_active()->face(1)->vertex(0);
    Assert(mesh.begin_active()->face(0)->boundary_id() == 0,
           dealii::ExcInvalidState());
    Assert(mesh.last_active()->face(1)->boundary_id() == 1,
           dealii::ExcInvalidState());
    boundary_conditions[0] = solution.value(x_first);
    boundary_conditions[1] = solution.value(x_last);
    TransportMmsTest::Test(solution);
  }
};

TEST_P(TransportMms1DTest, ManufacturedCosine) {
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 64, 0, 1);
  dealii::Functions::CosineFunction<dim> solution;
  Test(solution);
}

INSTANTIATE_TEST_CASE_P(FEDegree, TransportMms1DTest, ::testing::Range(0, 4));

}  // namespace

}  // namespace aether::sn