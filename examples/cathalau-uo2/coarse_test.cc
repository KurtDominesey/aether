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
    this->mesh.refine_global(1);
    dealii::FE_DGQ<dim_> fe(1);
    this->dof_handler.initialize(mesh, fe);
  }
};

TEST_P(CathalauCoarseTest, UniformFissionSource) {
  // SHEM-361 to CASMO-70
  const std::vector<int> g_maxes = {
      8, 11, 13, 17, 22, 25, 29, 32, 36, 39, 42, 48, 52, 55, 58, 60, 64, 67, 71,
      85, 112, 133, 152, 170, 200, 222, 275, 279, 287, 294, 296, 300, 305, 309,
      311, 313, 314, 316, 317, 318, 320, 321, 322, 323, 324, 326, 329, 332, 333,
      335, 336, 337, 338, 339, 340, 342, 344, 346, 347, 348, 349, 350, 351, 352,
      353, 354, 356, 357, 358, 360, 361};
  this->CompareCoarse(1, 50, 1e-8, true, 1000, 1e-8, g_maxes, this->materials);
}

INSTANTIATE_TEST_CASE_P(Fuel, CathalauCoarseTest,
                        ::testing::Values("uo2", "mox43"));

}  // namespace cathalau