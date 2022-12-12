#include <chrono>
#include <regex>

#include <hdf5.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/hdf5.h>
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
#include "base/lapack_full_matrix.h"
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

static const Lwr lwr1(0);  // case 1: unrodded
static const Lwr lwr2(1);  // case 2: rodded
static const Fbr fbr1(0);  // case 1: unrodded
static const Fbr fbr2(1);  // case 2: half-rodded
static const PinC5G7 pin_uo2_1("uo2", 1);
static const PinC5G7 pin_uo2_2("uo2", 2);
static const PinC5G7 pin_uo2_3("uo2", 3);
// static const PinC5G7 pin_mox("mox43");
static const std::vector<const Benchmark2D1D*> benchmarks{
  &lwr1, &lwr2, 
  // &fbr1, &fbr2, 
  &pin_uo2_1, &pin_uo2_2, &pin_uo2_3
};

enum MgDim : int {
  BOTH, ONE, TWO
};

namespace H5 = dealii::HDF5;

const std::string dir_out = "out/";

const int fe_degree = 1;

///////////////////////
// 3D Test (Ref. Soln.)
///////////////////////

class Test3D : public ::testing::TestWithParam<const Benchmark2D1D*> {
 protected:
  const Benchmark2D1D &bm = *GetParam();
  const aether::Mgxs mgxs = bm.create_mgxs();
  const int num_groups = mgxs.num_groups;
  dealii::Triangulation<3> mesh;
  dealii::DoFHandler<3> dof_handler;
  std::vector<std::vector<dealii::BlockVector<double>>> bc;

  void SetUp() override {
    bm.create_mesh3(mesh);
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
  dealii::GridOut grid_out;
  std::ofstream grid_file("out/mesh3.vtu");
  grid_out.write_vtu(mesh, grid_file);
  aether::sn::QPglc<3> quadrature(4, 4);
  bc.resize(num_groups, std::vector<dealii::BlockVector<double>>(1,
      dealii::BlockVector<double>(
          quadrature.size(),
          dof_handler.get_fe().dofs_per_cell
      )
  ));
  aether::sn::FixedSourceProblem<3> prob(dof_handler, quadrature, mgxs, bc);
  dealii::ReductionControl control_wg(250, 1e-6, 1e-2);
  dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg,
      dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData(17));
  auto print_wg = [] (
      const unsigned int it, const double val, const dealii::Vector<double>&) 
      -> dealii::SolverControl::State {
    std::cout << "WG " << it << ": " << val << "\n";
    return dealii::SolverControl::State::success;
  };
  solver_wg.connect(print_wg);
  aether::sn::FixedSourceGS fixed_src_gs(prob.fixed_source, solver_wg);
  dealii::BlockVector<double> src(
      num_groups, quadrature.size()*dof_handler.n_dofs());
  dealii::BlockVector<double> phi(num_groups, dof_handler.n_dofs());
  for (int g = 0; g < num_groups; ++g) {
    std::map<unsigned int, const std::unique_ptr<dealii::Function<3>>> funcs;
    std::map<unsigned int, const dealii::Function<3>*> func_ptrs;
    for (int j = 0; j < 1 /*mgxs.num_materials*/; ++j) {
      double chi = mgxs.chi[g][j];
      if (chi == 0)
        continue;
      // funcs.emplace(j, bm.create_source3(chi));
      funcs.emplace(j, 
          std::make_unique<dealii::Functions::ConstantFunction<3>>(chi));
    }
    for (auto it = funcs.begin(); it != funcs.end(); ++it) {
      func_ptrs[it->first] = it->second.get();
    }
    dealii::VectorTools::interpolate_based_on_material_id(
        dealii::MappingQGeneric<3>(fe_degree), dof_handler, func_ptrs, 
        phi.block(g));
    prob.m2d.vmult(src.block(g), phi.block(g));
  }
  dealii::BlockVector<double> flux(src.get_block_indices());
  dealii::BlockVector<double> uncollided(src.get_block_indices());
  for (int g = 0; g < num_groups; ++g)
    prob.fixed_source.within_groups[g].transport.vmult(
        uncollided.block(g), src.block(g), false);
  dealii::SolverControl control(50, 1e-6*uncollided.l2_norm());
  dealii::SolverGMRES<dealii::BlockVector<double>> solver(control,
      dealii::SolverGMRES<dealii::BlockVector<double>>::AdditionalData(7));
  solver.solve(prob.fixed_source, flux, uncollided, fixed_src_gs);
  // fixed_src_gs.vmult(flux, uncollided);
  // Save angular flux, plot scalar flux
  H5::File h5_out(dir_out+test_name()+".h5", H5::File::FileAccessMode::create);
  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  for (int g = 0; g < num_groups; ++g) {
    const std::string name_dataset = "g" + std::to_string(g+1);
    h5_out.write_dataset(name_dataset, flux.block(g));
    prob.d2m.vmult(phi.block(g), flux.block(g));
    data_out.add_data_vector(phi.block(g), name_dataset);
  }
  data_out.build_patches();
  std::ofstream vtu_out(dir_out + test_name() + ".vtu");
  data_out.write_vtu(vtu_out);
}

INSTANTIATE_TEST_CASE_P(Benchmarks, Test3D, ::testing::ValuesIn(benchmarks),
    [](const testing::TestParamInfo<const Benchmark2D1D*>& info) {
  return info.param->to_string();
});


//////////////
// 2D/1D Tests
//////////////

template <int dim>
using DoF = std::tuple<
    dealii::Point<dim>, 
    typename dealii::DoFHandler<dim>::active_cell_iterator,
    unsigned int
>;

template <int dim>
struct DoFSorter {
  DoFSorter(const double tol=1e-6) : tol(tol) {}
  const double tol;

  bool operator()(const DoF<dim> &a, const DoF<dim> &b) const {
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

/**
 * Sort DoFs by z-, y-, and x-coordinates (in that order).
 * 
 * The returned numbering is the opposite of that which should be passed to
 * dealii::DoFHandler<dim>.renumber_dofs.
 */
template <int dim>
std::vector<unsigned int> sort_dofs(dealii::DoFHandler<dim> &dof_handler) {
  std::vector<unsigned int> renumbering(dof_handler.n_dofs());
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
  std::stable_sort(dofs.begin(), dofs.end(), DoFSorter<dim>());
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    renumbering[i] = std::get<2>(dofs[i]);
  return renumbering;
}

using Param2D1D = std::tuple<const Benchmark2D1D*, bool, int, bool>;

class Test2D1D : public ::testing::TestWithParam<Param2D1D> {
 protected:
  const int num_modes = 30;
  const Benchmark2D1D &bm = *std::get<0>(GetParam());
  const bool axial_polar = std::get<1>(GetParam());
  const int mg_dim = std::get<2>(GetParam());
  const bool is_minimax = std::get<3>(GetParam());
  const aether::Mgxs mgxs = bm.create_mgxs();
  const int num_groups = mgxs.num_groups;
  const int num_groups2 = (mg_dim == BOTH || mg_dim == TWO) ? num_groups : 1;
  const int num_groups1 = (mg_dim == BOTH || mg_dim == ONE) ? num_groups : 1;
  const std::vector<std::vector<int>> &materials = bm.get_materials();
  const int num_segments = materials.size();
  const int num_areas = materials[0].size();
  aether::Mgxs mgxs1{num_groups1, num_segments, 1};
  aether::Mgxs mgxs2{num_groups2, num_areas, 1};
  const aether::sn::QPglc<2> quadrature2{axial_polar ? 0 : 4, 4};
  const aether::sn::QPglc<1> q_gauss{4};
  const aether::sn::QForwardBackward q_fb;
  const aether::sn::QAngle<1> &quadrature1 = axial_polar
      ? static_cast<const aether::sn::QAngle<1>&>(q_gauss) 
      : static_cast<const aether::sn::QAngle<1>&>(q_fb);
  const aether::sn::QPglc<3> quadrature3{4, 4};
  const dealii::FE_DGQ<1> fe1{fe_degree};
  const dealii::FE_DGQ<2> fe2{fe_degree};
  const dealii::FE_DGQ<3> fe3{fe_degree};
  dealii::Triangulation<1> mesh1;
  dealii::Triangulation<2> mesh2;
  dealii::Triangulation<3> mesh3;
  dealii::DoFHandler<1> dof_handler1;
  dealii::DoFHandler<2> dof_handler2;
  dealii::DoFHandler<3> dof_handler3;
  std::vector<unsigned int> order1, order2, order3;
  dealii::MappingQGeneric<1> mapping1{fe_degree};
  dealii::MappingQGeneric<2> mapping2{fe_degree};
  const std::vector<std::vector<dealii::BlockVector<double>>> bc1{
    num_groups, std::vector<dealii::BlockVector<double>>{
      1, dealii::BlockVector<double>{quadrature1.size(), fe1.dofs_per_cell}
    }
  };
  const std::vector<std::vector<dealii::BlockVector<double>>> bc2{
    num_groups, std::vector<dealii::BlockVector<double>>{
      1, dealii::BlockVector<double>{quadrature2.size(), fe2.dofs_per_cell}
    }
  };
  
  void SetUp() override {
    if (num_groups == num_groups1)
      mgxs1.group_widths = mgxs.group_widths;
    else
      mgxs1.group_widths.assign(mgxs1.num_groups, 1.);
    if (num_groups == num_groups2)
      mgxs2.group_widths = mgxs.group_widths;
    else
      mgxs2.group_widths.assign(mgxs2.num_groups, 1.);
    if (mgxs1.group_widths.size() == mgxs2.group_widths.size()) {
      mgxs1.group_widths.assign(mgxs1.num_groups, 1.);
      mgxs2.group_widths.assign(mgxs2.num_groups, 1.);
    }
    AssertThrow(axial_polar == !quadrature1.is_degenerate(), 
        dealii::ExcInvalidState());
    AssertThrow(axial_polar == quadrature2.is_degenerate(), 
        dealii::ExcInvalidState());
    bm.create_mesh1(mesh1);
    bm.create_mesh2(mesh2);
    bm.create_mesh3(mesh3);
    dof_handler1.initialize(mesh1, fe1);
    dof_handler2.initialize(mesh2, fe2);
    dof_handler3.initialize(mesh3, fe3);
    AssertThrow(dof_handler3.n_dofs() == 
                dof_handler1.n_dofs()*dof_handler2.n_dofs(),
                dealii::ExcInvalidState());
    order1 = sort_dofs(dof_handler1);
    order2 = sort_dofs(dof_handler2);
    order3 = sort_dofs(dof_handler3);
  }
};

class QuadratureAligner {
 public:
  QuadratureAligner(const aether::sn::QAngle<3> &q3,
                    const aether::sn::QAngle<2> &q2,
                    const aether::sn::QAngle<1> &q1,
                    const double tol=1e-12) :
      q3(q3), q2(q2), q1(q1),
      qp(q3.get_tensor_basis()[0]), qa(q3.get_tensor_basis()[1]), 
      halfpol(qp.size()/2), tol(tol), 
      cache(q3.size(), std::make_pair(q3.size(), q3.size())) {};

  const std::pair<int, int>& operator()(const int n3) const {
    auto [n1, n2] = cache[n3];
    if (n1 == q3.size() || n2 == q3.size()) {  //  invalid, not in cache
      const dealii::Point<2> &a3 = q3.angle(n3);
      int np = n3 % qp.size();
      int na = n3 / qp.size();
      if (q2.is_degenerate()) {
        n1 = np;
        n2 = na;
        AssertThrow(!q1.is_degenerate(), 
                    dealii::ExcMessage("One quadrature must not be degenerate"));
        AssertThrow(std::abs(a3[0]-q1.angle(n1)[0]) < tol, 
                    dealii::ExcMessage("Different polar angles"));
        AssertThrow(std::abs(a3[1]-q2.angle(n2)[1]) < tol, 
                    dealii::ExcMessage("Different azimuthal angles"));
      } else {
        n1 = np >= halfpol;
        n2 = na * halfpol + (n1 ? np - halfpol : halfpol - 1 - np);
        AssertThrow(q1.is_degenerate(), 
                    dealii::ExcMessage("One quadrature must be degenerate"));
        AssertThrow(a3[0] * q1.angle(n1)[0] > 0, 
                    dealii::ExcMessage("Opposite polar orientations"));
        AssertThrow(std::abs(std::abs(a3[0])-q2.angle(n2)[0]) < tol &&
                    std::abs(a3[1]-q2.angle(n2)[1]) < tol,
                    dealii::ExcMessage("Different angles"));
      }
      AssertIndexRange(n1, q1.size());
      AssertIndexRange(n2, q2.size());
      cache[n3] = std::make_pair(n1, n2);
    }
    return cache[n3];
  };

 protected:
  const aether::sn::QAngle<3> &q3;  // 3D quadrature
  const aether::sn::QAngle<2> &q2;  // 2D quadrature
  const aether::sn::QAngle<1> &q1;  // 1D quadrature
  const dealii::Quadrature<1> &qp;  // Polar basis of 3D quadrature
  const dealii::Quadrature<1> &qa;  // Azimuthal basis of 3D quadrature
  const int halfpol;
  const double tol;
  mutable std::vector<std::pair<int, int>> cache;
};

/**
 * Matrix pre-/post-processor for computing the SVD in the L2 norm.
 */
template <int dim>
class SvdL2PP {
 public:
  SvdL2PP(const dealii::DoFHandler<dim> &dof_handler,
          const std::vector<aether::sn::CellMatrices<dim>> &matrices,
          const std::vector<double> &q_weights,
          const std::vector<double> &u_widths) :
          dof_handler(dof_handler), 
          q_weights_sqrt(q_weights),
          u_widths_sqrt(u_widths) {
    for (double &weight : q_weights_sqrt)
      weight = std::sqrt(weight);
    for (double &width : u_widths_sqrt)
      width = std::sqrt(width);
    mass_cho.resize(matrices.size());
    for (int i = 0; i < matrices.size(); ++i)
      mass_cho[i].cholesky(matrices[i].mass);
    mass_cho_inv = mass_cho;
    for (int i = 0; i < matrices.size(); ++i)
      mass_cho_inv[i].gauss_jordan();
  }

  void preprocess(dealii::LAPACKFullMatrix_<double> &matrix, bool t=false) {
    process(matrix, t, true);
  }

  void postprocess(dealii::LAPACKFullMatrix_<double> &matrix, bool t=false) {
    process(matrix, t, false);
  }

 protected:
  void process(dealii::LAPACKFullMatrix_<double> &matrix, bool t, bool before) {
    const bool which = (dim == 1) != t;
    const int num_vecs = which ? matrix.m() : matrix.n();
    const int dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
    dealii::Vector<double> vec_c(dofs_per_cell);
    for (int v = 0; v < num_vecs; ++v) {
      for (int g = 0; g < u_widths_sqrt.size(); ++g) {
        int gg = g * q_weights_sqrt.size() * dof_handler.n_dofs();
        for (int n = 0; n < q_weights_sqrt.size(); ++n) {
          int nn = n * dof_handler.n_dofs() + gg;
          for (int c = 0; c < mass_cho.size(); ++c) {
            int cc = c * dofs_per_cell + nn;
            for (int i = 0; i < dofs_per_cell; ++i) {
              double &el = which ? matrix(v, cc+i) : matrix(cc+i, v);
              vec_c[i] = el;
              el = 0;
            }
            for (int i = 0; i < dofs_per_cell; ++i) {
              double &el = which ? matrix(v, cc+i) : matrix(cc+i, v);
              for (int j = 0; j < dofs_per_cell; ++j) {
                el += vec_c[j] * (before
                    ? mass_cho[c][i][j] * q_weights_sqrt[n] / u_widths_sqrt[g]
                    : mass_cho_inv[c][i][j] / q_weights_sqrt[n] * u_widths_sqrt[g]
                );
              }
            }
          }
        }
      }
    }
  }

  const dealii::DoFHandler<dim> &dof_handler;
  std::vector<double> q_weights_sqrt;
  std::vector<double> u_widths_sqrt;
  std::vector<dealii::FullMatrix<double>> mass_cho;
  std::vector<dealii::FullMatrix<double>> mass_cho_inv;
};

TEST_P(Test2D1D, Svd) {
  dealii::BlockVector<double> psi(
      num_groups, quadrature3.size()*dof_handler3.n_dofs());
  dealii::BlockVector<double> phi(num_groups, dof_handler3.n_dofs());
  aether::sn::DiscreteToMoment<3> d2m(quadrature3);
  // Load 3D ref. soln.
  std::string name_h5 = "BenchmarksTest3DFixedSource";
  name_h5 += bm.to_string();
  H5::File file_h5(dir_out+name_h5+".h5", H5::File::FileAccessMode::open);
  for (int g = 0; g < num_groups; ++g) {
    const std::string name_g = "g" + std::to_string(g+1);
    psi.block(g) = file_h5.open_dataset(name_g).read<dealii::Vector<double>>();
    d2m.vmult(phi.block(g), psi.block(g));
  }
  // Make some useful objects for later
  const QuadratureAligner q_align(quadrature3, quadrature2, quadrature1);
  aether::sn::Transport<1> transport1(dof_handler1, quadrature1);
  aether::sn::Transport<2> transport2(dof_handler2, quadrature2);
  aether::pgd::sn::Transport<3> transport3(dof_handler3, quadrature3);
  std::ofstream csv_out(dir_out+test_name()+".csv");
  csv_out << std::setprecision(std::numeric_limits<double>::max_digits10)
          << std::scientific
          << "err_psi,err_phi\n";
  if (num_groups1 > 1 && num_groups2 > 1) {  
    // Do groupwise SVD
    AssertDimension(num_groups1, num_groups2);
    AssertDimension(num_groups1, num_groups);
    std::vector<double> u_width(1, 1.);  // doesn't matter for groupwise SVD
    SvdL2PP svd_pp1(dof_handler1, transport1.cell_matrices, 
                    quadrature1.get_weights(), u_width);
    SvdL2PP svd_pp2(dof_handler2, transport2.cell_matrices, 
                    quadrature2.get_weights(), u_width);
    dealii::Table<2, double> errs_psi(num_groups, num_modes+1);
    dealii::Table<2, double> errs_phi(num_groups, num_modes+1);
    const int size2 = quadrature2.size() * dof_handler2.n_dofs();
    const int size1 = quadrature1.size() * dof_handler1.n_dofs();
    const bool t = size2 < size1;  // transpose
    for (int g = 0; g < num_groups; ++g) {
      // Make "snapshot" matrix
      // (For a m x n matrix, if m < n, the call to BLAS (gesdd) segfaults.)
      dealii::LAPACKFullMatrix_<double> psi_rect(t ? size1 : size2,
                                                 t ? size2 : size1);
      for (int n = 0; n < quadrature3.size(); ++n) {
        auto [n1, n2] = q_align(n);
        n1 *= dof_handler1.n_dofs();
        n2 *= dof_handler2.n_dofs();
        int nn = n * dof_handler3.n_dofs();
        for (int i = 0; i < dof_handler1.n_dofs(); ++i) {
          int ii = i * dof_handler2.n_dofs();
          for (int j = 0; j < dof_handler2.n_dofs(); ++j) {
            psi_rect(t ? n1+order1[i] : n2+order2[j], 
                     t ? n2+order2[j] : n1+order1[i]) =
                psi.block(g)[nn+order3[ii+j]];
          }
        }
      }
      svd_pp1.preprocess(psi_rect, t);
      svd_pp2.preprocess(psi_rect, t);
      std::cout << "do svd\n";
      psi_rect.compute_svd(/*thin svd*/'S');
      std::cout << "did svd\n";
      dealii::LAPACKFullMatrix_<double> basis2 = 
          t ? psi_rect.get_svd_vt() : psi_rect.get_svd_u();
      dealii::LAPACKFullMatrix_<double> basis1 = 
          t ? psi_rect.get_svd_u() : psi_rect.get_svd_vt();
      svd_pp1.postprocess(basis1, t);
      svd_pp2.postprocess(basis2, t);
      for (int m = 0; m <= num_modes; ++m) {
        errs_phi(g, m) = transport3.inner_product(phi.block(g), phi.block(g));
        errs_psi(g, m) = transport3.inner_product(psi.block(g), psi.block(g));
        if (m == num_modes)
          continue;
        for (int n = 0; n < quadrature3.size(); ++n) {
          auto [n1, n2] = q_align(n);
          n1 *= dof_handler1.n_dofs();
          n2 *= dof_handler2.n_dofs();
          int nn = n * dof_handler3.n_dofs();
          for (int i = 0; i < dof_handler1.n_dofs(); ++i) {
            int ii = i * dof_handler2.n_dofs();
            for (int j = 0; j < dof_handler2.n_dofs(); ++j) {
              const double v = 
                  psi_rect.singular_value(m) *
                  basis1(t ? n1+order1[i] : m, t ? m : n1+order1[i]) *
                  basis2(t ? m : n2+order2[j], t ? n2+order2[j] : m);
              psi.block(g)[nn+order3[ii+j]] -= v;
              phi.block(g)[order3[ii+j]] -= quadrature3.weight(n) * v;
            }
          }
        }
      }
    }
    for (int m = 0; m <= num_modes; ++m) {
      double err_psi = 0;
      double err_phi = 0;
      for (int g = 0; g < num_groups; ++g) {
        err_psi += errs_psi(g, m);
        err_phi += errs_phi(g, m);
      }
      err_psi = std::sqrt(err_psi);
      err_phi = std::sqrt(err_phi);
      csv_out << err_psi << "," << err_phi << "\n";
    }
    csv_out.close();
    return;
  }
  // Make "snapshot" matrix (for non-groupwise SVD)
  const int size2 = num_groups2 * quadrature2.size() * dof_handler2.n_dofs();
  const int size1 = num_groups1 * quadrature1.size() * dof_handler1.n_dofs();
  const bool t = size2 < size1;  // transpose
  // For a m x n matrix, if m < n, the call to BLAS (gesdd) segfaults.
  dealii::LAPACKFullMatrix_<double> psi_rect(t ? size1 : size2, 
                                             t ? size2 : size1);
  for (int g = 0; g < num_groups; ++g) {
    int g1 = num_groups1 > 1 ? g : 0;
    int g2 = num_groups2 > 1 ? g : 0;
    int gg1 = g1 * quadrature1.size() * dof_handler1.n_dofs();
    int gg2 = g2 * quadrature2.size() * dof_handler2.n_dofs();
    for (int n = 0; n < quadrature3.size(); ++n) {
      auto [n1, n2] = q_align(n);
      n1 = n1 * dof_handler1.n_dofs() + gg1;
      n2 = n2 * dof_handler2.n_dofs() + gg2;
      int nn = n * dof_handler3.n_dofs();
      for (int i = 0; i < dof_handler1.n_dofs(); ++i) {
        int ii = i * dof_handler2.n_dofs();
        for (int j = 0; j < dof_handler2.n_dofs(); ++j) {
          psi_rect(t ? n1+order1[i] : n2+order2[j], 
                   t ? n2+order2[j] : n1+order1[i]) =
              psi.block(g)[nn+order3[ii+j]];
        }
      }
    }
  }
  SvdL2PP svd_pp1(dof_handler1, transport1.cell_matrices, 
                  quadrature1.get_weights(), mgxs1.group_widths);
  SvdL2PP svd_pp2(dof_handler2, transport2.cell_matrices, 
                  quadrature2.get_weights(), mgxs2.group_widths);
  svd_pp1.preprocess(psi_rect, t);
  svd_pp2.preprocess(psi_rect, t);
  std::cout << "do svd\n";
  psi_rect.compute_svd(/*thin svd*/'S');
  std::cout << "did svd\n";
  dealii::LAPACKFullMatrix_<double> basis2 = 
      t ? psi_rect.get_svd_vt() : psi_rect.get_svd_u();
  dealii::LAPACKFullMatrix_<double> basis1 = 
      t ? psi_rect.get_svd_u() : psi_rect.get_svd_vt();
  svd_pp1.postprocess(basis1, t);
  svd_pp2.postprocess(basis2, t);
  // Compute error
  for (int m = 0; m <= num_modes; ++m) {
    std::cout << "singular value " << m << ": " 
              << psi_rect.singular_value(m) << "\n";
    double err_psi = 0;
    double err_phi = 0;
    for (int g = 0; g < num_groups; ++g) {
      int du = mgxs.group_widths[g]; // lethargy width
      err_phi += transport3.inner_product(phi.block(g), phi.block(g)) / du;
      err_psi += transport3.inner_product(psi.block(g), psi.block(g)) / du;
      if (m == num_modes)
        continue;
      int g1 = num_groups1 == 1 ? 0 : g;
      int g2 = num_groups2 == 1 ? 0 : g;
      g1 *= quadrature1.size() * dof_handler1.n_dofs();
      g2 *= quadrature2.size() * dof_handler2.n_dofs();
      for (int n = 0; n < quadrature3.size(); ++n) {
        auto [n1, n2] = q_align(n);
        n1 = n1 * dof_handler1.n_dofs() + g1;
        n2 = n2 * dof_handler2.n_dofs() + g2;
        int nn = n * dof_handler3.n_dofs();
        for (int i = 0; i < dof_handler1.n_dofs(); ++i) {
          int ii = i * dof_handler2.n_dofs();
          for (int j = 0; j < dof_handler2.n_dofs(); ++j) {
            const double v = 
                psi_rect.singular_value(m) *
                basis1(t ? n1+order1[i] : m, t ? m : n1+order1[i]) *
                basis2(t ? m : n2+order2[j], t ? n2+order2[j] : m);
            psi.block(g)[nn+order3[ii+j]] -= v;
            phi.block(g)[order3[ii+j]] -= quadrature3.weight(n) * v;
          }
        }
      }
    }
    err_psi = std::sqrt(err_psi);
    err_phi = std::sqrt(err_phi);
    csv_out << err_psi << "," << err_phi << "\n";
  }
  csv_out.close();
}

TEST_P(Test2D1D, Pgd) {
  dealii::GridOut grid_out;
  dealii::GridOutFlags::Svg svg;
  svg.coloring = dealii::GridOutFlags::Svg::Coloring::material_id;
  svg.margin = false;
  svg.label_cell_index = false;
  svg.label_level_number = false;
  svg.label_level_subdomain_id = false;
  svg.label_material_id = false;
  svg.label_subdomain_id = false;
  svg.draw_colorbar = false;
  svg.draw_legend = false;
  grid_out.set_flags(svg);
  std::ofstream grid_file(dir_out + test_name() + "-mesh2.svg");
  grid_out.write_svg(mesh2, grid_file);
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
  for (int g = 0; g < num_groups1; ++g) {
    double intensity = 1;
    if (num_groups1 == num_groups) {
      intensity *= mgxs.chi[g][0];
      if (num_groups2 == num_groups)
        intensity = std::sqrt(intensity);
    }
    // const dealii::ScalarFunctionFromFunctionObject<1> func1(
    //     [intensity] (const dealii::Point<1> &p, unsigned int=0) {
    //       return uniform_source<1>(intensity, p);});
    const dealii::Functions::ConstantFunction<1> func1(intensity);
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
    // const dealii::ScalarFunctionFromFunctionObject<2> func2(
    //     [intensity] (const dealii::Point<2> &p, unsigned int=0) {
    //       return uniform_source<2>(intensity, p);});
    const dealii::Functions::ConstantFunction<2> func2(intensity);
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
  // std::vector<std::vector<int>> materials{{0, 1, 2}, {1, 1, 2}};
  aether::pgd::sn::Nonlinear2D1D nonlinear2D1D(fs1, fs2, materials, mgxs,
      num_groups1 == num_groups2);
  fs1.is_minimax = is_minimax;
  fs2.is_minimax = is_minimax;
  const double tol = 1e-2;
  std::vector<std::chrono::duration<double>> runtimes(num_modes+1);
  runtimes[0] = std::chrono::duration<double>::zero();
  for (int m = 0; m < num_modes; ++m) {
    std::cout << "mode " << m << "\n";
    const auto start = std::chrono::steady_clock::now();
    nonlinear2D1D.enrich();
    for (int nl = 0; nl < 10; ++nl) {
      double r = nonlinear2D1D.iter();
      if (r < tol)
        break;
    }
    const auto end = std::chrono::steady_clock::now();
    runtimes[m+1] = end - start;
    // nonlinear2D1D.reweight();
  }
  // Expand solution
  aether::pgd::sn::Transport<3> transport3(dof_handler3, quadrature3);
  aether::sn::DiscreteToMoment<3> d2m(quadrature3);
  dealii::BlockVector<double> psi_ref(
      num_groups, quadrature3.size()*dof_handler3.n_dofs());
  dealii::BlockVector<double> phi_ref(num_groups, dof_handler3.n_dofs());
  // Compute error against ref. soln.
  std::string name_ref = "BenchmarksTest3DFixedSource";
  name_ref += bm.to_string();
  H5::File h5_ref(dir_out+name_ref+".h5", H5::File::FileAccessMode::open);
  for (int g = 0; g < num_groups; ++g) {
    const std::string name_g = "g" + std::to_string(g+1);
    psi_ref.block(g) = 
        h5_ref.open_dataset(name_g).read<dealii::Vector<double>>();
    d2m.vmult(phi_ref.block(g), psi_ref.block(g));
  }
  dealii::BlockVector<double> phi(phi_ref);
  const QuadratureAligner q_align(quadrature3, quadrature2, quadrature1);
  std::ofstream csv_out(dir_out+test_name()+".csv");
  csv_out << std::setprecision(std::numeric_limits<double>::max_digits10)
          << std::scientific
          << "runtime,err_psi,err_phi\n";
  for (int m = 0; m <= num_modes; ++m) {
    double err_psi = 0;
    double err_phi = 0;
    for (int g = 0; g < num_groups; ++g) {
      int du = mgxs.group_widths[g]; // lethargy width
      err_phi += 
          transport3.inner_product(phi_ref.block(g), phi_ref.block(g)) / du;
      err_psi += 
          transport3.inner_product(psi_ref.block(g), psi_ref.block(g)) / du;
      if (m == num_modes)
        continue;
      int g1 = num_groups1 == 1 ? 0 : g;
      int g2 = num_groups2 == 1 ? 0 : g;
      for (int i = 0; i < dof_handler1.n_dofs(); ++i) {
        int ii = i * dof_handler2.n_dofs();
        for (int j = 0; j < dof_handler2.n_dofs(); ++j) {
          phi_ref.block(g)[order3[ii+j]] -=
              fs1.prods[m].phi.block(g1)[order1[i]] *
              fs2.prods[m].phi.block(g2)[order2[j]];
        }
      }
      for (int n = 0; n < quadrature3.size(); ++n) {
        auto [n1, n2] = q_align(n);
        n1 *= dof_handler1.n_dofs();
        n2 *= dof_handler2.n_dofs();
        int nn = n * dof_handler3.n_dofs();
        for (int i = 0; i < dof_handler1.n_dofs(); ++i) {
          int ii = i * dof_handler2.n_dofs();
          for (int j = 0; j < dof_handler2.n_dofs(); ++j) {
            psi_ref.block(g)[nn+order3[ii+j]] -=
                fs1.prods[m].psi.block(g1)[n1+order1[i]] *
                fs2.prods[m].psi.block(g2)[n2+order2[j]];
          }
        }
      }
    }
    err_psi = std::sqrt(err_psi);
    err_phi = std::sqrt(err_phi);
    csv_out << runtimes[m].count() << "," << err_psi << "," << err_phi << "\n";
  }
  csv_out.close();
  phi -= phi_ref;  // ROM phi = FOM phi - (FOM phi - ROM phi)
  // Plot flux and errors
  dealii::BlockVector<double> psi_err(num_groups, dof_handler3.n_dofs());
  dealii::DataOut<3> data_out3;
  data_out3.attach_dof_handler(dof_handler3);
  for (int g = 0; g < num_groups; ++g) {
    const std::string suffix = "-g"+std::to_string(g+1);
    data_out3.add_data_vector(phi.block(g), "phi"+suffix);
    data_out3.add_data_vector(phi_ref.block(g), "phi"+suffix+"-err");
    d2m.vmult(psi_err.block(g), psi_ref.block(g));
    data_out3.add_data_vector(psi_err.block(g), "psi"+suffix+"-err");
  }
  data_out3.build_patches();
  std::ofstream vtu_out3(dir_out + test_name() + "-3D.vtu");
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
    ::testing::ValuesIn(benchmarks), ::testing::Bool(), ::testing::Range(0, 3), 
    ::testing::Values(false)  // just Galerkin, no Minimax for now
  ),
  [] (const testing::TestParamInfo<Param2D1D>& info) {
    std::string name;
    name += std::get<0>(info.param)->to_string();
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