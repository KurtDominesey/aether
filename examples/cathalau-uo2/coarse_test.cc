#include "cathalau_test.cc"
#include "../coarse_test.h"

namespace cathalau {

class CathalauCoarseTest : public CathalauTest,
                           public CoarseTest<dim_, qdim_>,
                           public ::testing::WithParamInterface<std::string> {
 protected:
  void SetUp() override {
    materials[2] = this->GetParam();
    group_structure = "SHEM-361";
    CathalauTest::SetUp();
    refine_azimuthal(this->mesh, 2);
    refine_radial(this->mesh, 2, this->max_levels);
    this->PrintMesh();
    dealii::FE_DGQ<dim_> fe(1);
    this->dof_handler.initialize(mesh, fe);
  }
  const std::vector<int> g_maxes = {  // SHEM-361 to CASMO-70
      9, 12, 14, 18, 23, 26, 30, 33, 37, 40, 43, 49, 53, 56, 59, 61, 65, 68,
      72, 86, 113, 134, 153, 171, 201, 223, 276, 280, 288, 295, 297, 301, 306,
      310, 312, 314, 315, 317, 318, 319, 321, 322, 323, 324, 325, 327, 330,
      333, 334, 336, 337, 338, 339, 340, 341, 343, 345, 347, 348, 349, 350,
      351, 352, 353, 354, 355, 357, 358, 359, 361};
};

TEST_P(CathalauCoarseTest, UniformFissionSource) {
  this->CompareCoarse(50, 50, 1e-4, true, 50, 1e-8, this->g_maxes, this->materials, {}, false);
}

// Running this method more than once causes a seg fault in collapse_spectra.
// Debugger doesn't give many clues--unsure why it only works on the first go.
// Workaround is to run each test individually. TODO: fix it properly
TEST_P(CathalauCoarseTest, Criticality) {
  const int num_modes = 50;
  const int iters_nl = 50;
  const double tol_nl = 1e-4;
  const bool do_update = true;
  const int iters_fom = 250;
  const double tol_fom = 1e-8;
  // galerkin (do_minimax is false)
  // this->CompareCoarse(
  //     num_modes, iters_nl, tol_nl, do_update, iters_fom, tol_fom, 
  //     this->g_maxes, this->materials, {},
  //     /*precomputed_full*/true, /*precomputed_cp*/true, /*precomputed_ip*/true, 
  //     /*should_write_mgxs*/true,  /*do_eigenvalue*/true, /*do_minimax*/false);
  // minimax
  this->CompareCoarse(
      50, iters_nl, tol_nl, do_update, iters_fom, tol_fom, 
      this->g_maxes, this->materials, {},
      /*precomputed_full*/true, /*precomputed_cp*/true, /*precomputed_ip*/true, 
      /*should_write_mgxs*/false,  /*do_eigenvalue*/true, /*do_minimax*/true);
  // this->CompareCoarse(50, 50, 1e-4, true, 250, 1e-8, this->g_maxes, 
  //                     this->materials, {},
  //                     /*precomputed_full*/false, /*precomputed_cp*/false, 
  //                     /*precomputed_ip*/false, /*should_write_mgxs*/true, 
  //                     /*do_eigenvalue*/true, do_minimax);

}

INSTANTIATE_TEST_CASE_P(Fuel, CathalauCoarseTest,
                        ::testing::Values("uo2", "mox43"));

}  // namespace cathalau