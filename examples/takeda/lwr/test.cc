#include <regex>

#include <deal.II/base/mpi.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>

#include "base/mgxs.h"
#include "sn/quadrature.h"
#include "sn/quadrature_lib.h"
#include "sn/fixed_source_problem.h"
#include "sn/fixed_source.h"
#include "sn/fixed_source_gs.h"
#include "pgd/sn/transport.h"
#include "pgd/sn/fixed_source_2D1D.h"
#include "pgd/sn/nonlinear_2D1D.h"

#include "mesh.h"
#include "mgxs.h"

#include "gtest/gtest.h"

int main (int argc, char **argv) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

namespace takeda::lwr {

enum Benchmark : int {
  LWR = 1,
  FBR_SMALL,
  FBR_HET_Z,
  FBR_HEX_Z
};

enum MgDim  : int {
  BOTH, ONE, TWO
};

const int fe_degree = 1;
const int num_groups = 2;
const int num_areas = 3;
const int num_segments = 2; 

class Test3D : public ::testing::TestWithParam<bool> {
 protected:
  dealii::Triangulation<3> mesh;
  dealii::DoFHandler<3> dof_handler;
  std::vector<std::vector<dealii::BlockVector<double>>> bc;

  void SetUp() override {
    create_mesh(mesh);
    dealii::FE_DGQ<3> fe(fe_degree);
    dof_handler.initialize(mesh, fe);
  }
};

std::string test_name() {
  const testing::TestInfo* const info =
    testing::UnitTest::GetInstance()->current_test_info();
  std::string name = info->test_case_name();
  name += info->name();
  return std::regex_replace(name, std::regex("/"), "");
}

template <int dim>
double uniform_source(double val, const dealii::Point<dim> &p) {
  for (int i = 0; i < dim; ++i)
    if (p[i] > 15)
      return 0;
  return std::pow(val, dim);
}

template <int dim>
double cosine_source(double val, const dealii::Point<dim> &p) {
  double src = val;
  for (int i = 0; i < dim; ++i)
    if (p[i] > 15)
      return 0;
    else
      src *= std::cos(p[i]);
  return std::pow(val, dim);
}

TEST_P(Test3D, FixedSource) {
  auto mgxs = create_mgxs(GetParam());
  aether::sn::QPglc<3> quadrature(4, 4);
  bc.resize(num_groups, std::vector<dealii::BlockVector<double>>(1,
      dealii::BlockVector<double>(
          quadrature.size(),
          dof_handler.get_fe().dofs_per_cell
      )
  ));
  aether::sn::FixedSourceProblem<3> prob(dof_handler, quadrature, mgxs, bc);
  dealii::SolverControl control_wg(100, 1e-2);
  dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg,
      dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData(32));
  aether::sn::FixedSourceGS fixed_src_gs(prob.fixed_source, solver_wg);
  dealii::BlockVector<double> src(
      num_groups, quadrature.size()*dof_handler.n_dofs());
  dealii::BlockVector<double> phi(num_groups, dof_handler.n_dofs());
  for (int g = 0; g < num_groups; ++g) {
    double intensity = mgxs.chi[g][0];
    const dealii::ScalarFunctionFromFunctionObject<3> func(
        [intensity] (const dealii::Point<3> &p, unsigned int=0) {
            return uniform_source<3>(intensity, p);});
    std::map<dealii::types::material_id, const dealii::Function<3>*> funcs;
    funcs[0] = &func;
    dealii::VectorTools::interpolate_based_on_material_id(
        dealii::MappingQGeneric<3>(fe_degree), dof_handler, funcs, 
        phi.block(g));
    prob.m2d.vmult(src.block(g), phi.block(g));
  }
  dealii::BlockVector<double> flux(src.get_block_indices());
  dealii::BlockVector<double> uncollided(src.get_block_indices());
  for (int g = 0; g < num_groups; ++g)
    prob.fixed_source.within_groups[g].transport.vmult(
        uncollided.block(g), src.block(g), false);
  fixed_src_gs.vmult(flux, uncollided);
  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  for (int g = 0; g < num_groups; ++g) {
    prob.d2m.vmult(phi.block(g), flux.block(g));
    data_out.add_data_vector(phi.block(g), "g"+std::to_string(g+1));
  }
  data_out.build_patches();
  std::ofstream vtu_out("out/" + test_name() + ".vtu");
  data_out.write_vtu(vtu_out);
}

INSTANTIATE_TEST_CASE_P(Rod, Test3D, ::testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
  return info.param ? "Rodded" : "Unrodded";
});

using Param2D1D = std::tuple<bool, bool, int, bool>;

class Test2D1D : public ::testing::TestWithParam<Param2D1D> {
 protected:
  dealii::Triangulation<1> mesh1;
  dealii::Triangulation<2> mesh2;
  dealii::DoFHandler<1> dof_handler1;
  dealii::DoFHandler<2> dof_handler2;
  std::vector<std::vector<dealii::BlockVector<double>>> bc1;
  std::vector<std::vector<dealii::BlockVector<double>>> bc2;

  void SetUp() override {
    create_mesh(mesh1);
    create_mesh(mesh2);
    dealii::FE_DGQ<1> fe1(1);
    dealii::FE_DGQ<2> fe2(1);
    dof_handler1.initialize(mesh1, fe1);
    dof_handler2.initialize(mesh2, fe2);
  }
};

template <int dim>
using DoF = std::tuple<
    dealii::Point<dim>, 
    typename dealii::DoFHandler<dim>::active_cell_iterator,
    unsigned int
>;

template <int dim>
struct CompareDoF {
  bool operator()(const DoF<dim> &a, const DoF<dim> &b) {
    const double tol = 1e-12;
    const dealii::Point<dim> &ca = std::get<1>(a)->center();
    const dealii::Point<dim> &cb = std::get<1>(b)->center();
    for (int d = dim-1; d >= 0; --d) {
      // compare support points
      if (std::abs(std::get<0>(a)[d] - std::get<0>(b)[d]) > tol)
        return std::get<0>(a)[d] < std::get<0>(b)[d];
      // compare centers
      if (std::abs(ca[d] - cb[d]) > tol)
        return ca[d] < cb[d];
    }
    AssertThrow(false, dealii::ExcInvalidState());
  }
};

template <int dim>
void sort_dofs(dealii::DoFHandler<dim> &dof_handler, 
               std::vector<unsigned int> &renumbering) {
  dealii::MappingQGeneric<dim> mapping(fe_degree);
  std::vector<dealii::Point<dim>> points(dof_handler.n_dofs());
  dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler, points);
  std::vector<DoF<dim>> dofs;
  dofs.reserve(dof_handler.n_dofs());
  std::vector<unsigned int> dof_indices(dof_handler.get_fe().dofs_per_cell);
  for (auto &cell : dof_handler.active_cell_iterators()) {
    cell->get_dof_indices(dof_indices);
    for (unsigned int i : dof_indices) {
      dofs.emplace_back(points[i], cell, i);
    }
  }
  CompareDoF<dim> comparator;
  std::stable_sort(dofs.begin(), dofs.end(), comparator);
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    renumbering[std::get<2>(dofs[i])] = i;
}

TEST_P(Test2D1D, FixedSource) {
  const bool is_rodded = std::get<0>(GetParam());
  const bool axial_polar = std::get<1>(GetParam());
  const bool is_minimax = std::get<3>(GetParam());
  const int mg_dim = std::get<2>(GetParam());
  const int num_groups2 = (mg_dim == BOTH || mg_dim == TWO) ? num_groups : 1;
  const int num_groups1 = (mg_dim == BOTH || mg_dim == ONE) ? num_groups : 1;
  std::vector<unsigned int> order1(dof_handler1.n_dofs());
  std::vector<unsigned int> order2(dof_handler2.n_dofs());
  sort_dofs<1>(dof_handler1, order1);
  sort_dofs<2>(dof_handler2, order2);
  std::vector<unsigned int> reorder1(dof_handler1.n_dofs());
  std::vector<unsigned int> reorder2(dof_handler2.n_dofs());
  for (int i = 0; i < dof_handler1.n_dofs(); ++i)
    reorder1[order1[i]] = i;
  for (int i = 0; i < dof_handler2.n_dofs(); ++i)
    reorder2[order2[i]] = i;
  auto mgxs = create_mgxs(is_rodded);
  aether::Mgxs mgxs1(num_groups1, num_segments, 1);
  aether::Mgxs mgxs2(num_groups2, num_areas, 1);
  const aether::sn::QPglc<2> quadrature2(axial_polar ? 0 : 4, 4);
  const aether::sn::QPglc<1> q_gauss(4);
  const aether::sn::QForwardBackward q_fb;
  const aether::sn::QAngle<1> &quadrature1 = axial_polar
      ? static_cast<const aether::sn::QAngle<1>&>(q_gauss) 
      : static_cast<const aether::sn::QAngle<1>&>(q_fb);
  AssertThrow(axial_polar == !quadrature1.is_degenerate(), 
      dealii::ExcInvalidState());
  AssertThrow(axial_polar == quadrature2.is_degenerate(), 
      dealii::ExcInvalidState());
  bc1.resize(num_groups1, std::vector<dealii::BlockVector<double>>(1,
      dealii::BlockVector<double>(
          quadrature1.size(),
          dof_handler1.get_fe().dofs_per_cell
      )
  ));
  bc2.resize(num_groups2, std::vector<dealii::BlockVector<double>>(1,
      dealii::BlockVector<double>(
          quadrature2.size(),
          dof_handler2.get_fe().dofs_per_cell
      )
  ));
  aether::sn::FixedSourceProblem<1, 1, aether::pgd::sn::Transport<1>, 
                                       aether::pgd::sn::TransportBlock<1> >
      problem1(dof_handler1, quadrature1, mgxs1, bc1);
  aether::sn::FixedSourceProblem<2, 2, aether::pgd::sn::Transport<2>,
                                       aether::pgd::sn::TransportBlock<2> > 
      problem2(dof_handler2, quadrature2, mgxs2, bc2);
  std::vector<dealii::BlockVector<double>> srcs1, srcs2;
  srcs1.emplace_back(num_groups1, quadrature1.size()*dof_handler1.n_dofs());
  srcs2.emplace_back(num_groups2, quadrature2.size()*dof_handler2.n_dofs());
  dealii::Vector<double> phi1(dof_handler1.n_dofs());
  dealii::Vector<double> phi2(dof_handler2.n_dofs());
  dealii::MappingQGeneric<1> mapping1(fe_degree);
  dealii::MappingQGeneric<2> mapping2(fe_degree);
  for (int g = 0; g < num_groups1; ++g) {
    double intensity = 1;
    if (num_groups1 == num_groups) {
      intensity *= mgxs.chi[g][0];
      if (num_groups2 == num_groups)
        intensity = std::sqrt(intensity);
    }
    const dealii::ScalarFunctionFromFunctionObject<1> func1(
        [intensity] (const dealii::Point<1> &p, unsigned int=0) {
          return uniform_source<1>(intensity, p);});
    std::map<dealii::types::material_id, const dealii::Function<1>*> funcs1;
    funcs1[0] = &func1;
    dealii::VectorTools::interpolate_based_on_material_id(
        mapping1, dof_handler1, funcs1, phi1);
    problem1.m2d.vmult(srcs1[0].block(g), phi1);
  }
  for (int g = 0; g < num_groups2; ++g) {
    double intensity = 1;
    if (num_groups2 == num_groups) {
      intensity *= mgxs.chi[g][0];
      if (num_groups1 == num_groups)
        intensity = std::sqrt(intensity);
    }
    const dealii::ScalarFunctionFromFunctionObject<2> func2(
        [intensity] (const dealii::Point<2> &p, unsigned int=0) {
          return uniform_source<2>(intensity, p);});
    std::map<dealii::types::material_id, const dealii::Function<2>*> funcs2;
    funcs2[0] = &func2;
    dealii::VectorTools::interpolate_based_on_material_id(
        mapping2, dof_handler2, funcs2, phi2);
    problem2.m2d.vmult(srcs2[0].block(g), phi2);
  }
  // Run the 2D/1D PGD simulation
  aether::pgd::sn::FixedSource2D1D<1> fs1(
      problem1.fixed_source, srcs1, problem1.transport, mgxs1);
  aether::pgd::sn::FixedSource2D1D<2> fs2(
      problem2.fixed_source, srcs2, problem2.transport, mgxs2);
  std::vector<std::vector<int>> materials{{0, 1, 2}, {1, 1, 2}};
  aether::pgd::sn::Nonlinear2D1D nonlinear2D1D(fs1, fs2, materials, mgxs,
      num_groups1 == num_groups2);
  fs1.is_minimax = is_minimax;
  fs2.is_minimax = is_minimax;
  const int num_modes = 5;
  for (int m = 0; m < num_modes; ++m) {
    std::cout << "mode " << m << "\n";
    nonlinear2D1D.enrich();
    for (int nl = 0; nl < 10; ++nl) {
      double r = nonlinear2D1D.iter();
    }
  }
  // Expand solution
  dealii::Triangulation<3> mesh3;
  create_mesh(mesh3);
  dealii::FE_DGQ<3> fe3(fe_degree);
  dealii::DoFHandler<3> dof_handler3;
  dof_handler3.initialize(mesh3, fe3);
  // Sort all the DoFs into a known order
  std::vector<unsigned int> order3(dof_handler3.n_dofs());
  sort_dofs<3>(dof_handler3, order3);
  dof_handler3.renumber_dofs(order3);
  dealii::BlockVector<double> phi(num_groups, 
      dof_handler1.n_dofs()*dof_handler2.n_dofs());
  for (int m = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g) {
      int g1 = num_groups1 == 1 ? 0 : g;
      int g2 = num_groups2 == 1 ? 0 : g;
      for (int i = 0; i < dof_handler1.n_dofs(); ++i) {
        int ii = i * dof_handler2.n_dofs();
        for (int j = 0; j < dof_handler2.n_dofs(); ++j) {
          phi.block(g)[ii+j] += fs1.prods[m].phi.block(g1)[reorder1[i]] *
                                fs2.prods[m].phi.block(g2)[reorder2[j]];
        }
      }
    }
  }
  // Plot flux
  dealii::DataOut<3> data_out3;
  data_out3.attach_dof_handler(dof_handler3);
  for (int g = 0; g < num_groups; ++g) {
    data_out3.add_data_vector(phi.block(g), "g"+std::to_string(g+1));
  }
  data_out3.build_patches();
  std::ofstream vtu_out3("out/" + test_name() + "-3D.vtu");
  data_out3.write_vtu(vtu_out3);
  // Plot modes
  dealii::DataOut<1> data_out1;
  dealii::DataOut<2> data_out2;
  data_out1.attach_dof_handler(dof_handler1);
  data_out2.attach_dof_handler(dof_handler2);
  for (int m = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups1; ++g)
      data_out1.add_data_vector(fs1.prods[m].phi.block(g), 
          "m"+std::to_string(m+1)+"g"+std::to_string(g+1));
    for (int g = 0; g < num_groups2; ++g)
      data_out2.add_data_vector(fs2.prods[m].phi.block(g), 
          "m"+std::to_string(m+1)+"g"+std::to_string(g+1));
  }
  data_out1.build_patches();
  data_out2.build_patches();
  std::ofstream vtu_out1("out/" + test_name() + "-1D.vtu");
  std::ofstream vtu_out2("out/" + test_name() + "-2D.vtu");
  data_out1.write_vtu(vtu_out1);
  data_out2.write_vtu(vtu_out2);
}

INSTANTIATE_TEST_CASE_P(Params, Test2D1D, ::testing::Combine(
    ::testing::Bool(), ::testing::Bool(), ::testing::Range(0, 3), 
    ::testing::Values(false)  // just Galerkin, no Minimax for now
  ),
  [] (const testing::TestParamInfo<Param2D1D>& info) {
    std::string name;
    name += "Rod";
    name += std::get<0>(info.param) ? "Y" : "N";
    name += "Pol";
    name += std::get<1>(info.param) ? "1" : "2";
    name += "Mg";
    switch (std::get<2>(info.param)) {
      case BOTH: name += "B"; break;
      case ONE: name += "1"; break;
      case TWO: name += "2"; break;
      default: AssertThrow(false, dealii::ExcInvalidState());
    }
    name += std::get<3>(info.param) ? "Mmx" : "Gal";
    return name;
  }
);

}  // namespace takeda::lwr