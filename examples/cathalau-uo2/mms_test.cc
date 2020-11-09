#include "cathalau_test.cc"
#include "../mms_test.h"

namespace cathalau {

class CathalauMmsTest : public CathalauTest,
                        public MmsTest<dim_, qdim_>, 
                        public ::testing::WithParamInterface<int> {
 protected:
  void SetUp() override {
    CathalauTest::SetUp(true);
    dealii::FE_DGQ<dim_> fe(WithParamInterface<int>::GetParam());
    this->dof_handler.initialize(this->mesh, fe);
  };
};

TEST_P(CathalauMmsTest, FullOrder) {
  double factor = dealii::numbers::PI_2 / this->pitch;
  MmsTest<dim_, qdim_>::TestFullOrder(4, 3000, 1e-6, 2 * factor);
}

TEST_P(CathalauMmsTest, Pgd) {
  double factor = dealii::numbers::PI_2 / this->pitch;
  MmsTest<dim_, qdim_>::TestPgd(4, 50, 2 * factor);
}

INSTANTIATE_TEST_CASE_P(FEDegree, CathalauMmsTest, ::testing::Range(0, 3));

}  // namespace cathalau
