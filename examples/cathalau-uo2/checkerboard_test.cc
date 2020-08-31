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

enum Pattern {
  CHECKERBOARD,
  STRIPES
};

class CheckerboardTest : public CathalauTest,
                         public CoarseTest<dim_, qdim_>,
                         public ::testing::WithParamInterface<Pattern> {
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
    const std::string suffix = "/mgxs-" + group_structure + ".h5";
    read_mgxs(mgxs_uo2, dir+"uo2"+suffix, "294K", materials_uo2, true);
    read_mgxs(mgxs_mox, dir+"mox43"+suffix, "294K", materials_mox, true);
    materials.insert(
        materials.end(), materials_mox.begin(), materials_mox.end());
    mgxs = std::make_unique<Mgxs>(num_groups, materials.size(), num_legendre);
    for (int g = 0; g < num_groups; ++g) {
      for (int j = 0; j < materials_uo2.size(); ++j) {
        mgxs->total[g][j] = mgxs_uo2.total[g][j];
        mgxs->chi[g][j] = mgxs_uo2.chi[g][j];
        mgxs->nu_fission[g][j] = mgxs_uo2.nu_fission[g][j];
        for (int gp = 0; gp < num_groups; ++gp)
          mgxs->scatter[g][gp][j] = mgxs_uo2.scatter[g][gp][j];
      }
      for (int j = 0; j < materials_mox.size(); ++j) {
        int jj = materials_uo2.size() + j;
        mgxs->total[g][jj] = mgxs_mox.total[g][j];
        mgxs->chi[g][jj] = mgxs_mox.chi[g][j];
        mgxs->nu_fission[g][jj] = mgxs_mox.nu_fission[g][j];
        for (int gp = 0; gp < num_groups; ++gp)
          mgxs->scatter[g][gp][jj] = mgxs_mox.scatter[g][gp][j];
      }
    }
    for (int g = 0; g < mgxs->group_structure.size(); ++g)
      mgxs->group_structure[g] = mgxs_uo2.group_structure[g];
    quadrature = QPglc<qdim_>(4, 8);
    AssertDimension(regions.size(), radii.size()+1);
    std::vector<int> regions_mox = regions;
    for (int r = 0; r < regions_mox.size(); ++r)
      regions_mox[r] += materials_uo2.size();
    const int mani_id_trans = 1;
    const int mani_id_left = 2;
    const int mani_id_right = 3;
    dealii::Triangulation<dim_> octant_left;
    dealii::Triangulation<dim_> octant_right;
    const Pattern pattern = this->GetParam();
    switch (pattern) {
      case CHECKERBOARD:
        mesh_eighth_pincell(octant_left, radii, pitch, regions, 
                            mani_id_trans, mani_id_left);
        mesh_eighth_pincell_ul(octant_right, radii, pitch, regions_mox,
                               mani_id_trans, mani_id_right);
        break;
      case STRIPES:
        mesh_symmetric_quarter_pincell(octant_left, radii, pitch, regions, 
                                       mani_id_trans, mani_id_left);
        mesh_symmetric_quarter_pincell(octant_right, radii, pitch, regions_mox,
                                      mani_id_trans, mani_id_right);
        break;
    }
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
    std::vector<int> max_levels_half = max_levels;
    max_levels.insert(
        max_levels.end(), max_levels_half.begin(), max_levels_half.end());
    refine_azimuthal(mesh, 2);
    refine_radial(mesh, 2, max_levels);
    this->PrintMesh();
    dealii::FE_DGQ<dim_> fe(1);
    dof_handler.initialize(mesh, fe);
    std::cout << "return SetUp\n";
  }

  void RunInfiniteMedium(dealii::Vector<double> &spectrum,
                         const Mgxs &mgxs, const std::vector<double> &volumes) {
    const int num_groups = mgxs.total.size();
    const int num_materials = mgxs.total[0].size();
    AssertDimension(spectrum.size(), num_groups);
    AssertDimension(volumes.size(), num_materials);
    // Assemble linear system
    spectrum = 0;
    dealii::FullMatrix<double> matrix(num_groups);
    for (int j = 0; j < num_materials; ++j) {
      std::cout << volumes[j] << " ";
      for (int g = 0; g < num_groups; ++g) {
        spectrum[g] += mgxs.chi[g][j] * volumes[j];
        matrix[g][g] += mgxs.total[g][j] * volumes[j];
        for (int gp = 0; gp < num_groups; ++gp) {
          matrix[g][gp] -= mgxs.scatter[g][gp][j] * volumes[j];
        }
      }
    }
    std::cout << std::endl;
    matrix.gauss_jordan();
    // Power iterations
    dealii::Vector<double> emission(spectrum.size());
    double k = 1;
    double emitted = 0;
    double emitted_last = 0;
    double power = 0;
    double power_last = 0;
    for (int i = 0; i < 100; ++i) {
      emission = 0;
      power = 0;
      for (int j = 0; j < num_materials; ++j) {
        double power_j = 0;
        for (int g = 0; g < num_groups; ++g)
          power_j += mgxs.nu_fission[g][j] * spectrum[g] * volumes[j];
        for (int g = 0; g < num_groups; ++g)
          emission[g] += power_j * mgxs.chi[g][j];
        power += power_j;
        // std::cout << power << std::endl;
      }
      emitted_last = emitted;
      for (int g = 0; g < num_groups; ++g)
        emitted += emission[g];
      // std::cout << emitted << std::endl;
      if (i > 0) {
        // k = (emitted / emitted_last) * k;
        double k_last = k;
        k = (power / power_last) * k;
        // std::cout << std::abs(k - k_last) * 1e5 << " pcm" << std::endl;
      }
      power_last = power;
      emission /= k;
      matrix.vmult(spectrum, emission);
      // std::cout << k << std::endl;
    }
    std::cout << k << std::endl;
    return;
  }

  void CompareCoarse(const int num_modes,
                     const int max_iters_nonlinear,
                     const double tol_nonlinear,
                     const bool do_update,
                     const int max_iters_fullorder,
                     const double tol_fullorder,
                     const std::vector<int> &g_maxes,
                     const std::vector<std::string> &materials,
                     const bool unequal_powers) {
    std::vector<double> factors(materials.size(), 1.);
    if (unequal_powers) {
      // run infinite medium problem
      std::vector<double> volumes;
      this->SetVolumes(volumes);
      for (int j = 0; j < materials_mox.size(); ++j)
        volumes[j+materials_uo2.size()] = volumes[j];
      const int num_groups = mgxs->total.size();
      dealii::Vector<double> spectrum(num_groups);
      RunInfiniteMedium(spectrum, *mgxs, volumes);
      int j_uo2 = 2;
      int j_mox = materials_uo2.size() + j_uo2;
      double emitted_uo2 = 0;
      double emitted_mox = 0;
      for (int g = 0; g < num_groups; ++g) {
        emitted_uo2 += mgxs->nu_fission[g][j_uo2] * spectrum[g] * volumes[j_uo2];
        emitted_mox += mgxs->nu_fission[g][j_mox] * spectrum[g] * volumes[j_mox];
      }
      double mox_ratio = emitted_mox / emitted_uo2;
      std::cout << "RATIO " << mox_ratio << std::endl;
      factors[j_mox] = mox_ratio;
      // return;
    }
    CoarseTest<dim_, qdim_>::CompareCoarse(
        num_modes, max_iters_nonlinear, tol_nonlinear, do_update, 
        max_iters_fullorder, tol_fullorder, g_maxes, materials, factors, true);
    const int num_legendre = 1;
    // const int num_groups = mgxs->total.size();
    const int num_groups_coarse = g_maxes.size();
    // consistent p and inconsistent p
    std::vector<std::string> corrections = {"", "_ip"};
    for (std::string &correction : corrections) {
      std::cout << correction << "\n";
      // Combine mgxs's
      std::cout << "Combine mgxs's\n";
      // Mgxs mgxs_uo2(num_groups_coarse, materials_uo2.size(), num_legendre);
      // Mgxs mgxs_mox(num_groups_coarse, materials_mox.size(), num_legendre);
      const std::string base = "Fuel_CathalauCoarseTestUniformFissionSource_";
      const std::string suffix = correction + "_mgxs-gold.h5";
      // read_mgxs(mgxs_uo2, base+"uo2"+suffix, "294K", materials_uo2);
      // read_mgxs(mgxs_mox, base+"mox43"+suffix, "294K", materials_mox);
      const Mgxs mgxs_uo2 = 
          read_mgxs(base+"uo2"+suffix, "294K", materials_uo2);
      const Mgxs mgxs_mox = 
          read_mgxs(base+"mox43"+suffix, "294K", materials_mox);
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
      this->RunFullOrder(flux_coarse, CoarseTest<dim_, qdim_>::source_coarse, 
                         problem_coarse, max_iters_fullorder, tol_fullorder);
      // Post-process
      std::cout << "Post-process\n";
      TransportType &transport = problem_coarse.transport;
      std::cout << "d2m\n";
      // DiscreteToMoment<qdim_> d2m(quadrature);
      DiscreteToMoment<qdim_> &d2m = problem_coarse.d2m;
      std::vector<double> _;
      std::vector<double> l2_errors_coarse_sep_d_rel;
      std::vector<double> l2_errors_coarse_sep_m_rel;
      std::vector<double> l2_errors_coarse_sep_d_abs;
      std::vector<double> l2_errors_coarse_sep_m_abs;
      std::cout << "coarse_sep_d_abs\n";
      GetL2ErrorsCoarseDiscrete(l2_errors_coarse_sep_d_abs, flux_coarse, 
                                flux_coarsened, transport, false, table, 
                                "coarse_sep_d_abs"+correction, _);
      std::cout << "coarse_sep_m_abs\n";
      GetL2ErrorsCoarseMoments(l2_errors_coarse_sep_m_abs, flux_coarse, 
                              flux_coarsened, transport, d2m, false, table, 
                              "coarse_sep_m_abs"+correction, _);
      std::cout << "coarse_sep_d_rel\n";
      GetL2ErrorsCoarseDiscrete(l2_errors_coarse_sep_d_rel, flux_coarse, 
                                flux_coarsened, transport, true, table, 
                                "coarse_sep_d_rel"+correction, _);
      std::cout << "coarse_sep_m_rel\n";
      GetL2ErrorsCoarseMoments(l2_errors_coarse_sep_m_rel, flux_coarse, 
                              flux_coarsened, transport, d2m, true, table,
                              "coarse_sep_m_rel"+correction, _);
      this->PlotFlux(flux_coarse, d2m, group_structure_coarse, 
                     "coarse"+correction+"_dissimilar");
      this->PlotDiffAngular(flux_coarse, flux_coarsened, d2m,
                            "diff_angular_coarse"+correction+"_dissimilar");
      this->PlotDiffScalar(flux_coarse, flux_coarsened, d2m,
                           "diff_scalar_coarse"+correction+"_dissimilar");
      for (int g = 0; g < num_groups_coarse; ++g) {
        // table.add_value("flux_coarse_sep", flux_coarse.block(g).l2_norm());
        // table.add_value("flux_coarsened_2", flux_coarsened.block(g).l2_norm());
        table.add_value("source_coarse_2", source_coarse.block(g).l2_norm());
      }
      for (std::string key : 
          {"flux_coarse_sep", "flux_coarsened_2", "source_coarse_2"}) {
        table.set_scientific(key, true);
        table.set_precision(key, 16);
      }
    }
    std::cout << "printing table to file\n";
    this->WriteConvergenceTable(table);
    std::cout << "printed\n";
  }
};

// SHEM-361 to CASMO-70
const std::vector<int> g_maxes = {
    9, 12, 14, 18, 23, 26, 30, 33, 37, 40, 43, 49, 53, 56, 59, 61, 65, 68,
    72, 86, 113, 134, 153, 171, 201, 223, 276, 280, 288, 295, 297, 301, 306,
    310, 312, 314, 315, 317, 318, 319, 321, 322, 323, 324, 325, 327, 330,
    333, 334, 336, 337, 338, 339, 340, 341, 343, 345, 347, 348, 349, 350,
    351, 352, 353, 354, 355, 357, 358, 359, 361};

TEST_P(CheckerboardTest, EqualPowers) {
  this->CompareCoarse(50, 50, 1e-4, true, 50, 1e-8, g_maxes, this->materials, 
                      false);
}

TEST_P(CheckerboardTest, UnequalPowers) {
  this->CompareCoarse(50, 50, 1e-4, true, 50, 1e-8, g_maxes, this->materials, 
                      true);
}

INSTANTIATE_TEST_CASE_P(Pattern, CheckerboardTest,
                        ::testing::Values(CHECKERBOARD, STRIPES));

}