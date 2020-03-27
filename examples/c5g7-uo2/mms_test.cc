#include "c5g7_test.cc"
#include "../mms_test.h"

class C5G7MmsTest : public C5G7Test,
                    public MmsTest<dim_, qdim_>, 
                    public ::testing::WithParamInterface<int> {
 protected:
  void SetUp() override {
    C5G7Test::SetUp();
    dealii::FE_DGQ<dim_> fe(this->GetParam());
    this->dof_handler.initialize(this->mesh, fe);
  };
};

TEST_P(C5G7MmsTest, FullOrder) {
  double factor = dealii::numbers::PI_2 / this->pitch;
  this->TestFullOrder(3000, 1e-6, 2 * factor);
}

TEST_P(C5G7MmsTest, Pgd) {
  double factor = dealii::numbers::PI_2 / this->pitch;
  this->TestPgd(50, 2 * factor);
}

INSTANTIATE_TEST_CASE_P(FEDegree, C5G7MmsTest, ::testing::Range(0, 3));
