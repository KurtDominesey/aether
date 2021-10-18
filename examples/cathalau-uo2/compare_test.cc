#include "cathalau_test.cc"
#include "../compare_test.h"

namespace cathalau {

class CathalauCompareTest : public CathalauTest, 
                            public CompareTest<dim_, qdim_>,
                            public ::testing::WithParamInterface<
                                std::tuple<std::string, std::string, int>> {
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
  this->Compare(50, 50, 1e-4, 50, 1e-8, false, false, false, false);
}

TEST_P(CathalauCompareTest, WithUpdate) {
  this->Compare(50, 50, 1e-2, 300, 1e-10, true, /*do minimax*/false,
                /*precomputed_full*/true, /*precomputed_pgd*/true, 
                /*do_eigenvalue*/false, /*full_only*/false);
}

TEST_P(CathalauCompareTest, MinimaxWithUpdate) {
  this->Compare(50, 50, 1e-2, 300, 1e-10, true, /*do minimax*/true,
                /*precomputed_full*/true, /*precomputed_pgd*/true, 
                /*do_eigenvalue*/false, /*full_only*/false);
}

TEST_P(CathalauCompareTest, WithEigenUpdate) {
  this->Compare(50, 50, 1e-2, 300, 1e-10, true, /*do minimax*/false,
                /*precomputed_full*/true, /*precomputed_pgd*/false, 
                /*do_eigenvalue*/true, /*full_only*/false);
}

TEST_P(CathalauCompareTest, MinimaxWithEigenUpdate) {
  this->Compare(50, 50, 1e-2, 300, 1e-10, true, /*do minimax*/true,
                /*precomputed_full*/true, /*precomputed_pgd*/false, 
                /*do_eigenvalue*/true, /*full_only*/false);
}

INSTANTIATE_TEST_CASE_P(GroupStructure, CathalauCompareTest,
    ::testing::Combine(
    ::testing::Values("uo2", "mox43"),
    ::testing::Values(//"CASMO-8", "CASMO-16", "CASMO-25", "CASMO-40", 
                      "CASMO-70", "XMAS-172", "SHEM-361"
                      // ,"CCFE-709", "UKAEA-1102"
                      ),
    ::testing::Values(0)
    )
);

// INSTANTIATE_TEST_CASE_P(CasmoStructure, CathalauCompareTest, 
//     ::testing::Values("CASMO-8", "CASMO-16", "CASMO-25", "CASMO-40", "CASMO-70")
//     );

class CathalauCompareSubspaceTest : public CathalauCompareTest {};

TEST_P(CathalauCompareSubspaceTest, PgdEnergy) {
  const int num_modes_s = std::get<2>(this->GetParam());
  this->Compare(50, 50, 1e-2, 300, 1e-10, /*do update*/true, 
                /*do minimax*/false, /*precomputed_full*/true, 
                /*precompute_pgd*/true,  /*do_eigenvalue*/true, 
                /*full_only*/false, num_modes_s, /*guess_svd*/false, 
                /*guess_spatioangular*/true);
}

TEST_P(CathalauCompareSubspaceTest, PgdBoth) {
  const int num_modes_s = std::get<2>(this->GetParam());
  this->Compare(50, 50, 1e-2, 300, 1e-10, true, false, true, true, true, 
                false, num_modes_s, false, false);
}

TEST_P(CathalauCompareSubspaceTest, SvdEnergy) {
  const int num_modes_s = std::get<2>(this->GetParam());
  this->Compare(50, 50, 1e-2, 300, 1e-10, true, false, true, true, true,
                false, num_modes_s, true, true);
}

TEST_P(CathalauCompareSubspaceTest, SvdBoth) {
  const int num_modes_s = std::get<2>(this->GetParam());
  this->Compare(50, 50, 1e-2, 300, 1e-10, true, false, true, true, true,
                false, num_modes_s, true, false);
}

INSTANTIATE_TEST_CASE_P(GroupStructureModes, CathalauCompareSubspaceTest,
    ::testing::Combine(
    ::testing::Values("mox43"),
    ::testing::Values("CASMO-70"),
    ::testing::Values(5, 10, 20, 30, 40, 50)
    )
);

}  // namespace cathalau