#include "cathalau_test.cc"
#include "../mgxs_test.h"

namespace cathalau {

class CathalauMgxsTest : public CathalauTest,
                         public MgxsTest<dim_, qdim_>,
                         public ::testing::WithParamInterface<std::string> {
 protected:
  using CathalauTest::materials;

  void SetUp() override {
    materials[2] = this->GetParam();
    group_structure = "SHEM-361";
    CathalauTest::SetUp();
    this->mesh.refine_global(1);
    dealii::FE_DGQ<dim_> fe(1);
    this->dof_handler.initialize(this->mesh, fe);
  };

  void CompareMgxs(const int num_modes,
                   const int max_iters_nonlinear,
                   const double tol_nonlinear,
                   const int max_iters_fullorder,
                   const double tol_fullorder,
                   const bool do_update) {
    // SHEM-361 to CASMO-70
    std::vector<int> g_maxes = {
        8, 11, 13, 17, 22, 25, 29, 32, 36, 39, 42, 48, 52, 55, 58, 60, 64, 67,
        71, 85, 112, 133, 152, 170, 200, 222, 275, 279, 287, 294, 296, 300, 305,
        309, 311, 313, 314, 316, 317, 318, 320, 321, 322, 323, 324, 326, 329,
        332, 333, 335, 336, 337, 338, 339, 340, 342, 344, 346, 347, 348, 349,
        350, 351, 352, 353, 354, 356, 357, 358, 360, 361};
    AssertDimension(regions.size(), radii.size()+1);
    std::vector<double> volumes(materials.size());
    std::vector<double> areas(radii.size());  // ring areas
    for (int r = 0; r < radii.size(); ++r)
      areas[r]= dealii::numbers::PI * std::pow(radii[r], 2);
    volumes[regions.back()] = std::pow(2*pitch, 2) - areas.back();
    for (int r = 0; r < radii.size(); ++r)
      volumes[regions[r]] += areas[r] - (r == 0 ? 0 : areas[r-1]);
    MgxsTest<dim_, qdim_>::CompareMgxs(
        num_modes, max_iters_nonlinear, tol_nonlinear, max_iters_fullorder, 
        tol_fullorder, do_update, g_maxes, volumes);
  }
};

TEST_P(CathalauMgxsTest, ToCasmo70) {
  this->CompareMgxs(50, 100, 1e-8, 1000, 1e-8, true);
}

INSTANTIATE_TEST_CASE_P(Fuel, CathalauMgxsTest, 
                        ::testing::Values("uo2", "mox43"));

}  // namespace cathalau