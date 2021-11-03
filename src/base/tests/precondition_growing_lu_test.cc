#include "base/precondition_growing_lu.h"
#include "gtest/gtest.h"

namespace aether {

namespace {

class PreconditionGrowingLUTest : public ::testing::Test {
 protected:
  dealii::LAPACKFullMatrix_<double> a, b, c, d, eye;
  PreconditionGrowingLU<double> a_lu;
};

TEST_F(PreconditionGrowingLUTest, GrowThrice) {
  int block_size = 5;
  a.reinit(block_size);
  eye.reinit(block_size);
  for (int i = 0; i < block_size; ++i)
    a(i, i) = i + 1;
  a_lu.initialize(a);
  eye = a;
  a_lu.matrix.solve(eye);
  for (int i = 0; i < eye.m(); ++i)
    for (int j = 0; j < eye.n(); ++j)
      EXPECT_NEAR(eye(i, j), i == j, 1e-14);
  for (int num_blocks = 1; num_blocks <= 3; ++num_blocks) {
    int size = num_blocks * block_size;
    b.reinit(size, block_size);
    c.reinit(block_size, size);
    d.reinit(block_size);
    a.grow_or_shrink(a.m()+block_size);
    eye.reinit(eye.m()+block_size);
    int last = num_blocks * block_size;
    for (int bl = 0; bl <= num_blocks; ++bl) {
      int off = bl * block_size;
      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          if (bl < num_blocks) {
            b(off+i, j) = (bl+1) * (i+j+1);
            c(i, off+j) = -(bl+1) * (i+j+1);
            a(off+i, last+j) = b(off+i, j);
            a(last+i, off+j) = c(i, off+j);
          } else {
            d(i, j) = (i == j) * std::pow(i+2, 2);
            a(last+i, last+j) = d(i, j);
          }
        }
      }
    }
    a_lu.grow(b, c, d);
    eye = a;
    a_lu.matrix.solve(eye);
    for (int i = 0; i < eye.m(); ++i)
      for (int j = 0; j < eye.n(); ++j)
        EXPECT_NEAR(eye(i, j), i == j, 1e-14);
  }
}

}  // namespace

}  // namespace aether