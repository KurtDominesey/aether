#include "cathalau_test.cc"
#include "../compare_test.h"

namespace cathalau {

class CathalauCompareTest : public CathalauTest, 
                            public CompareTest<dim_, qdim_>,
                            public ::testing::WithParamInterface<
                                std::tuple<std::string, std::string>> {
 protected:
  void SetUp() override {
    materials[2] = std::get<0>(this->GetParam());
    group_structure = std::get<1>(this->GetParam());
    CathalauTest::SetUp();
    refine_azimuthal(this->mesh, 2);
    refine_radial(this->mesh, 2, this->max_levels);
    this->PrintMesh();
    dealii::FE_DGQ<dim_> fe(1);
    this->dof_handler.initialize(mesh, fe);
  }  
};

TEST_P(CathalauCompareTest, Progressive) {
  this->Compare(50, 50, 1e-4, 50, 1e-8, false, false, false);
}

TEST_P(CathalauCompareTest, WithUpdate) {
  this->Compare(50, 50, 1e-4, 50, 1e-8, true, false, false);
}

INSTANTIATE_TEST_CASE_P(GroupStructure, CathalauCompareTest,
    ::testing::Combine(
    ::testing::Values("uo2", "mox43"),
    ::testing::Values(//"CASMO-8", "CASMO-16", "CASMO-25", "CASMO-40", 
                      "CASMO-70", "XMAS-172", "SHEM-361"
                      // ,"CCFE-709", "UKAEA-1102"
                      )
    )
);

// INSTANTIATE_TEST_CASE_P(CasmoStructure, CathalauCompareTest, 
//     ::testing::Values("CASMO-8", "CASMO-16", "CASMO-25", "CASMO-40", "CASMO-70")
//     );

}  // namespace cathalau