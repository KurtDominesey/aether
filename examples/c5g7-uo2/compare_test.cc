#include "c5g7_test.cc"
#include "../compare_test.h"

class C5G7CompareTest : public C5G7Test, public CompareTest<dim_, qdim_> {
 protected:
  void SetUp() override {
    C5G7Test::SetUp();
    this->mesh.refine_global(2);
    dealii::FE_DGQ<dim_> fe(1);
    this->dof_handler.initialize(mesh, fe);
  }  
};

TEST_F(C5G7CompareTest, Progressive) {
  this->Compare(30, 50, 1e-4, 50, 1e-10, false, false, false, false);
}

TEST_F(C5G7CompareTest, WithUpdate) {
  this->Compare(30, 50, 1e-4, 50, 1e-10, true, false, false, false);
}