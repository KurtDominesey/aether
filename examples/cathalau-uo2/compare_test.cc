#include "cathalau_test.cc"
#include "../compare_test.h"

namespace cathalau {

class CathalauCompareTest : public CathalauTest, 
                            public CompareTest<dim_, qdim_>,
                            public ::testing::WithParamInterface<std::string> {
 protected:
  void SetUp() override {
    group_structure = this->GetParam();
    CathalauTest::SetUp();
    this->mesh.refine_global(1);
    dealii::FE_DGQ<dim_> fe(1);
    this->dof_handler.initialize(mesh, fe);
  }  
};

TEST_P(CathalauCompareTest, Progressive) {
  this->Compare(30, 50, 1e-6, 1000, 1e-6, false);
}

TEST_P(CathalauCompareTest, WithUpdate) {
  this->Compare(30, 50, 1e-6, 1000, 1e-6, true);
}

INSTANTIATE_TEST_CASE_P(GroupStructure, CathalauCompareTest, 
    ::testing::Values("CASMO-8", "CASMO-16", "CASMO-25", "CASMO-40", "CASMO-70", 
                      "XMAS-172", "SHEM-361"
                      // ,"CCFE-709", "UKAEA-1102"
                      ));

// INSTANTIATE_TEST_CASE_P(CasmoStructure, CathalauCompareTest, 
//     ::testing::Values("CASMO-8", "CASMO-16", "CASMO-25", "CASMO-40", "CASMO-70")
//     );

}  // namespace cathalau