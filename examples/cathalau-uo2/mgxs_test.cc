#include "cathalau_test.cc"
#include "../mgxs_test.h"

namespace cathalau {

class CathalauMgxsTest : public CathalauTest,
                         public MgxsTest<dim_, qdim_>,
                         public ::testing::WithParamInterface<std::string> {
 protected:
  using CathalauTest::materials;

  void SetUp() override {
    CathalauTest::SetUp();
    this->mesh.refine_global(2);
    dealii::FE_DGQ<dim_> fe(1);
    this->dof_handler.initialize(this->mesh, fe);
  };

  void CompareMgxs(const int num_modes,
                   const int max_iters_nonlinear,
                   const double tol_nonlinear,
                   const int max_iters_fullorder,
                   const double tol_fullorder,
                   const bool do_update) {
    std::vector<int> g_maxes;
    if (this->GetParam() == "CASMO-70")
      throw dealii::ExcInvalidState();
      // g_maxes = {24, 55, 65, 70};
      // g_maxes = {10, 14, 18, 24, 43, 55, 65, 70};
    else if (this->GetParam() == "XMAS-172")
      throw dealii::ExcInvalidState();
      // g_maxes = {37, 125, 150, 172};
      // g_maxes = {12, 20, 26, 37, 80, 125, 150, 172};
    else if (this->GetParam() == "SHEM-361")
      // g_maxes = {34, 302, 338, 361};
      // g_maxes = {12, 18, 24, 34, 85, 302, 338, 361};
      g_maxes = {
          2, 4, 5, 7, 8, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21, 23, 24, 26,
          26, 27, 29, 31, 34, 36, 36, 38, 39, 41, 42, 44, 45, 47, 48, 49, 51,
          55, 60, 64, 66, 73, 81, 85, 138, 160, 190, 208, 227, 248, 275, 289,
          293, 296, 300, 302, 305, 308, 312, 318, 321, 324, 328, 331, 335, 338,
          343, 347, 349, 352, 361};
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
  this->CompareMgxs(30, 100, 1e-6, 1000, 1e-6, true);
}

INSTANTIATE_TEST_CASE_P(GroupStructure, CathalauMgxsTest,
    ::testing::Values("SHEM-361"));

}  // namespace cathalau