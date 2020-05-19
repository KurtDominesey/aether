#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "cathalau_test.cc"
#include "../coarse_test.h"
#include "../compare_test.h"

namespace cathalau {

using namespace aether;
using namespace aether::sn;

class CheckerboardTest : public CathalauTest,
                         public CoarseTest<dim_, qdim_> {
 protected:
  std::vector<std::string> materials_uo2;
  std::vector<std::string> materials_mox;

  void SetUp() override {
    group_structure = "SHEM-361";
    materials_uo2 = materials;
    materials_mox = materials;
    materials_mox[2] = "mox43";
    const int hyphen = group_structure.find("-");
    const std::string num_groups_str = group_structure.substr(hyphen+1);
    const int num_groups = std::stoi(num_groups_str);
    const int num_legendre = 1;
    Mgxs mgxs_uo2(num_groups, materials_uo2.size(), num_legendre);
    Mgxs mgxs_mox(num_groups, materials_mox.size(), num_legendre);
    const std::string dir = "/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/";
    const std::string suffix = "/mgxs-" + group_structure + "-a.h5";
    read_mgxs(mgxs_uo2, dir+"uo2"+suffix, "294K", materials_uo2, true);
    read_mgxs(mgxs_mox, dir+"mox43"+suffix, "294K", materials_mox, true);
    materials.insert(
        materials.end(), materials_mox.begin(), materials_mox.end());
    mgxs = std::make_unique<Mgxs>(num_groups, materials.size(), num_legendre);
    for (int g = 0; g < num_groups; ++g) {
      for (int j = 0; j < materials_uo2.size(); ++j) {
        mgxs->total[g][j] = mgxs_uo2.total[g][j];
        mgxs->chi[g][j] = mgxs_uo2.chi[g][j];
        for (int gp = 0; gp < num_groups; ++gp)
          mgxs->scatter[g][gp][j] = mgxs_uo2.scatter[g][gp][j];
      }
      for (int j = 0; j < materials_mox.size(); ++j) {
        int jj = materials_uo2.size() + j;
        mgxs->total[g][jj] = mgxs_mox.total[g][j];
        mgxs->chi[g][jj] = mgxs_mox.chi[g][j];
        for (int gp = 0; gp < num_groups; ++gp)
          mgxs->scatter[g][gp][jj] = mgxs_mox.scatter[g][gp][j];
      }
    }
    for (int g = 0; g < mgxs->group_structure.size(); ++g)
      mgxs->group_structure[g] = mgxs_uo2.group_structure[g];
    quadrature = QPglc<qdim_>(4, 4);
    AssertDimension(regions.size(), radii.size()+1);
    const int mani_id_trans = 1;
    const int mani_id_left = 2;
    const int mani_id_right = 3;
    dealii::Triangulation<dim_> octant_left;
    dealii::Triangulation<dim_> octant_right;
    mesh_symmetric_quarter_pincell(octant_left, radii, pitch, regions, 
                                   mani_id_trans, mani_id_left);
    std::vector<int> regions_mox = regions;
    for (int r = 0; r < regions_mox.size(); ++r)
      regions_mox[r] += materials_uo2.size();
    mesh_symmetric_quarter_pincell(octant_right, radii, pitch, regions_mox,
                                   mani_id_trans, mani_id_right);
    dealii::GridTools::rotate(dealii::numbers::PI_2, octant_right);
    dealii::GridTools::shift(dealii::Point<dim_>(pitch*2, 0), octant_right);
    dealii::GridGenerator::merge_triangulations(
        octant_left, octant_right, mesh, 1e-12, true);
    dealii::TransfiniteInterpolationManifold<dim_> mani_trans;
    dealii::SphericalManifold<dim_> mani_left;
    dealii::SphericalManifold<dim_> mani_right(dealii::Point<dim_>(pitch*2, 0));
    mani_trans.initialize(mesh);
    mesh.set_manifold(mani_id_trans, mani_trans);
    mesh.set_manifold(mani_id_left, mani_left);
    mesh.set_manifold(mani_id_right, mani_right);
    set_all_boundaries_reflecting(mesh);
    mesh.refine_global(1);
    dealii::FE_DGQ<dim_> fe(1);
    dof_handler.initialize(mesh, fe);
    dealii::GridOut grid_out;
    std::ofstream file("checkerboard.svg");
    grid_out.write_svg(mesh, file);
    std::cout << "return SetUp\n";
  }

  void CompareCoarse(const int num_modes,
                     const int max_iters_nonlinear,
                     const double tol_nonlinear,
                     const bool do_update,
                     const int max_iters_fullorder,
                     const double tol_fullorder,
                     const std::vector<int> &g_maxes,
                     const std::vector<std::string> &materials) {
    CoarseTest<dim_, qdim_>::CompareCoarse(
        num_modes, max_iters_nonlinear, tol_nonlinear, do_update, 
        max_iters_fullorder, tol_fullorder, g_maxes, materials);
    const int num_legendre = 1;
    // const int num_groups = mgxs->total.size();
    const int num_groups_coarse = g_maxes.size();
    // Combine mgxs's
    std::cout << "Combine mgxs's\n";
    // Mgxs mgxs_uo2(num_groups_coarse, materials_uo2.size(), num_legendre);
    // Mgxs mgxs_mox(num_groups_coarse, materials_mox.size(), num_legendre);
    const std::string base = "Fuel_CathalauCoarseTestUniformFissionSource_";
    const std::string suffix = "_mgxs-gold.h5";
    // read_mgxs(mgxs_uo2, base+"uo2"+suffix, "294K", materials_uo2);
    // read_mgxs(mgxs_mox, base+"mox43"+suffix, "294K", materials_mox);
    const Mgxs mgxs_uo2 = read_mgxs(base+"uo2"+suffix, "294K", materials_uo2);
    const Mgxs mgxs_mox = read_mgxs(base+"mox43"+suffix, "294K", materials_mox);
    Mgxs mgxs_coarse(num_groups_coarse, materials.size(), num_legendre);
    bool nonzero = false;
    for (int g = 0; g < num_groups_coarse; ++g) {
      for (int j = 0; j < materials_uo2.size(); ++j) {
        mgxs_coarse.total[g][j] = mgxs_uo2.total[g][j];
        // mgxs->chi[g][j] = mgxs_uo2.chi[g][j];
        for (int gp = 0; gp < num_groups_coarse; ++gp) {
          mgxs_coarse.scatter[g][gp][j] = mgxs_uo2.scatter[g][gp][j];
          if (mgxs_uo2.scatter[g][gp][j] > 0)
            nonzero = true;
        }
      }
      for (int j = 0; j < materials_mox.size(); ++j) {
        int jj = materials_uo2.size() + j;
        mgxs_coarse.total[g][jj] = mgxs_mox.total[g][j];
        // mgxs->chi[g][jj] = mgxs_mox.chi[g][j];
        for (int gp = 0; gp < num_groups_coarse; ++gp) {
          mgxs_coarse.scatter[g][gp][jj] = mgxs_mox.scatter[g][gp][j];
          if (mgxs_mox.scatter[g][gp][j] > 0)
            nonzero = true;
        }
      }
    }
    AssertThrow(nonzero, dealii::ExcInvalidState());
    // Run coarse group
    std::cout << "Run coarse group\n";
    dealii::BlockVector<double> flux_coarse(
        num_groups_coarse, quadrature.size()*dof_handler.n_dofs());
    std::cout << num_groups_coarse << std::endl;
    std::cout << flux_coarse.size() << std::endl;
    std::cout << CoarseTest<dim_, qdim_>::source_coarse.size() << std::endl;
    std::vector<std::vector<dealii::BlockVector<double>>>
        boundary_conditions_coarse(num_groups_coarse);
    using TransportType = pgd::sn::Transport<dim_, qdim_>;
    FixedSourceProblem<dim_, qdim_, TransportType> problem_coarse(
          dof_handler, quadrature, mgxs_coarse, boundary_conditions_coarse);
    this->RunFullOrder(flux_coarse, CoarseTest<dim_, qdim_>::source_coarse, problem_coarse,
                       max_iters_fullorder, tol_fullorder);
    // Post-process
    std::cout << "Post-process\n";
    TransportType &transport = problem_coarse.transport;
    std::cout << "d2m\n";
    // DiscreteToMoment<qdim_> d2m(quadrature);
    DiscreteToMoment<qdim_> &d2m = problem_coarse.d2m;
    std::vector<double> l2_errors_coarse_sep_d_rel;
    std::vector<double> l2_errors_coarse_sep_m_rel;
    std::vector<double> l2_errors_coarse_sep_d_abs;
    std::vector<double> l2_errors_coarse_sep_m_abs;
    std::cout << "coarse_sep_d_abs\n";
    GetL2ErrorsCoarseDiscrete(l2_errors_coarse_sep_d_abs, flux_coarse, 
                              flux_coarsened, transport, false, table, 
                              "coarse_sep_d_abs");
    std::cout << "coarse_sep_m_abs\n";
    GetL2ErrorsCoarseMoments(l2_errors_coarse_sep_m_abs, flux_coarse, 
                             flux_coarsened, transport, d2m, false, table, 
                             "coarse_sep_m_abs");
    std::cout << "coarse_sep_d_rel\n";
    GetL2ErrorsCoarseDiscrete(l2_errors_coarse_sep_d_rel, flux_coarse, 
                              flux_coarsened, transport, true, table, 
                              "coarse_sep_d_rel");
    std::cout << "coarse_sep_m_rel\n";
    GetL2ErrorsCoarseMoments(l2_errors_coarse_sep_m_rel, flux_coarse, 
                             flux_coarsened, transport, d2m, true, table,
                             "coarse_sep_m_rel");
    // for (int g = 0; g < num_groups_coarse; ++g) {
    //   table.add_value("flux_coarse_sep", flux_coarse.block(g).l2_norm());
    //   // table.add_value("flux_coarsened_2", flux_coarsened.block(g).l2_norm());
    //   // table.add_value("source_coarse_2", source_coarse.block(g).l2_norm());
    // }
    // for (std::string key : 
    //     {"flux_coarse_sep"/*, "flux_coarsened_2", "source_coarse_2"*/}) {
    //   table.set_scientific(key, true);
    //   table.set_precision(key, 16);
    // }
    std::cout << "printing table to file\n";
    this->WriteConvergenceTable(table);
    std::cout << "printed\n";
  }
};

TEST_F(CheckerboardTest, UniformFissionSource) {
  // SHEM-361 to CASMO-70
  const std::vector<int> g_maxes = {
      9, 12, 14, 18, 23, 26, 30, 33, 37, 40, 43, 49, 53, 56, 59, 61, 65, 68,
      72, 86, 113, 134, 153, 171, 201, 223, 276, 280, 288, 295, 297, 301, 306,
      310, 312, 314, 315, 317, 318, 319, 321, 322, 323, 324, 325, 327, 330,
      333, 334, 336, 337, 338, 339, 340, 341, 343, 345, 347, 348, 349, 350,
      351, 352, 353, 354, 355, 357, 358, 359, 361};
  this->CompareCoarse(50, 50, 1e-8, true, 1000, 1e-8, g_maxes, this->materials);
}

}