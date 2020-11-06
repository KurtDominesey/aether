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
    refine_azimuthal(this->mesh, 2);
    refine_radial(this->mesh, 2, this->max_levels);
    this->PrintMesh();
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
        9, 12, 14, 18, 23, 26, 30, 33, 37, 40, 43, 49, 53, 56, 59, 61, 65, 68,
        72, 86, 113, 134, 153, 171, 201, 223, 276, 280, 288, 295, 297, 301, 306,
        310, 312, 314, 315, 317, 318, 319, 321, 322, 323, 324, 325, 327, 330,
        333, 334, 336, 337, 338, 339, 340, 341, 343, 345, 347, 348, 349, 350,
        351, 352, 353, 354, 355, 357, 358, 359, 361};
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
  this->CompareMgxs(50, 50, 1e-4, 50, 1e-8, true);
}

INSTANTIATE_TEST_CASE_P(Fuel, CathalauMgxsTest, 
                        ::testing::Values("uo2", "mox43"));

}  // namespace cathalau