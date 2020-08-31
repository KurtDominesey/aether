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
};

TEST_P(CathalauCoarseTest, UniformFissionSource) {
  // SHEM-361 to CASMO-70
  const std::vector<int> g_maxes = {
      9, 12, 14, 18, 23, 26, 30, 33, 37, 40, 43, 49, 53, 56, 59, 61, 65, 68,
      72, 86, 113, 134, 153, 171, 201, 223, 276, 280, 288, 295, 297, 301, 306,
      310, 312, 314, 315, 317, 318, 319, 321, 322, 323, 324, 325, 327, 330,
      333, 334, 336, 337, 338, 339, 340, 341, 343, 345, 347, 348, 349, 350,
      351, 352, 353, 354, 355, 357, 358, 359, 361};
  this->CompareCoarse(50, 50, 1e-4, true, 50, 1e-10, g_maxes, this->materials, {}, true);
}

INSTANTIATE_TEST_CASE_P(Fuel, CathalauCoarseTest,
                        ::testing::Values("uo2", "mox43"));

}  // namespace cathalau